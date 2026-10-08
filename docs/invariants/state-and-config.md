# SharedState, live config and credits

The evidence behind the rules in `.claude/rules/arch-state-and-config.md`: what each
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
Recorded by `SharedState::accept_origin_part` (was `record_origin_downloaded_shard`)
from BOTH origin-download paths — the auto-manage downloader and the admin "download
this part" handler; the second recorded nothing until it was added, which is the same
one-invariant-N-paths trap as everything else in this file. **Since 2026-10-04 an
origin download is recorded only when something CORROBORATES it** — a connected holder
that checked its copy holds these bytes, or the previous download of the part brought
the same ones (`uncorroborated_origin_parts`); otherwise it is deleted and fetched again
(FUTURE_WORK #217: one node's two downloads in a row came out as two different wrong
builds, each recorded here as the origin's; huggingface/huggingface_hub#3643 reports the
same shape — right size, a different hash per attempt). `huggingface::download_shard`
itself now syncs the file and checks it reads back as the bytes that arrived.
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
   The entries apply wherever the file is read, because `parse_config_file` is
   its one reader (until 2026-10-08 only the loader applied them, and the next
   dashboard save put the old value back — #242).
5. **Unknown keys warn, they do not fail.** `deny_unknown_fields` would refuse
   to start on a config mentioning a later release's key. `warn_unknown_keys_in`
   names the key and continues.

## `SharedState::release_request_state`

(2026-08-09) — clears the maps a
finished request leaves behind: `active_pipelines`, `active_traces`,
`request_holder_blacklist`, `peer_vram_commitments` and
`local_memory_refusals` (since then also `route_plan_overrides`,
`salvaged_replies` and `retained_activations`). They are keyed by request id and share one
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
one-invariant-N-paths defect again (`.claude/rules/arch-state-and-config.md`), in the one
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

**From the rules file (moved 2026-10-02):**

`credit::escrow` has three paths that change an escrow's status and then move a
balance — `release_escrow`, `refund_escrow`, `cleanup_expired`. **All three
leave the entry `Pending` and the balance untouched when the status write
fails**, and return `Err`. Two of them did; `release_escrow` warned and
reconciled anyway, which mints credits — `EscrowManager::new` re-inserts every
`Pending` entry at startup, so the settled escrow comes back claimable and the
expiry sweep refunds the full reservation. 500 in, 40 consumed, 560 out.

The trade-off direction is fixed: **lost-or-refunded, never double-paid.**

`Database::set_write_failure(Some(tree))` is how a persist failure is caused in
a test, armed at `with_write_table`. It is scoped to a TREE deliberately —
failing every write hides this class of bug, because the balance move reverts
itself and the books come out even.

→ `docs/invariants/state-and-config.md`

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



### The override half (2026-09-17, gotcha #633)

The predicate now takes the request's `swarm_route` override as a **required**
argument. `pretend_local_holds: "none"` means "plan as if this node held none of
this model", and it answered `x-swarm-route: local`, `x-swarm-peers: 0` —
byte-identical to the same request without the block. The instruction was
parsed, validated (a mistyped value still returned a 400 naming the field) and
then dropped, because this predicate decides whether the request ever reaches
the router and **the router is the only reader of the override**.

Auto-manage converges nodes on holding whole models — that is its job — so the
knob was inert on precisely the machines it was written for. Its own module doc
names that convergence as the reason it exists.

**`override_keeps_whole_model(pretend, shard_count)` is the pure decision**, a
truth table beside `local_fast_path_allowed`: `None`/`Everything` keep the fast
path, `Nothing` refuses it, and a range keeps it only when it covers every shard
of the model. A range with **no manifest to size it against refuses**, because
it cannot be shown to cover the model and the caller explicitly asked this node
to hold less. `exclude_nodes` is deliberately NOT consulted — leaving a peer out
of the candidate set says nothing about what WE hold, and the fast path asks no
peer for anything.

Three callers, each of which must now say what it means: the two dispatch paths
pass the request's override, and `api::admin_models::listing` passes `None`
because a listing answers for the node rather than for a request.

**The generalisable half.** This is the third feature this fast path has eaten
— #187 (a predicate and a getter disagreeing about which split entry),
#443 (the #442 delegation fix unreachable because the fast path always won on a
processor-only node), and now this. The section above already ends with the
sentence that would have caught it: *a feature behind a gate is only as
reachable as the gate's callers: test it from the API, not from the function.*
**When adding anything request-scoped, enumerate every early return between the
API edge and the code that reads it.** Validating the field at the edge proves
nothing: a 400 on a mistyped value is exactly what a knob that is then discarded
also produces.

Verified from the API, not the function: on a node holding the model in full,
unchanged without the block and genuinely answered by another machine with it
(`route: distributed`, `peers: 1`, a real remote node id).
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

## A partial config update builds on the FILE, not on what the daemon remembers

**Rule:** `.claude/rules/arch-state-and-config.md` § "A partial config update
builds on the FILE, not on what the daemon remembers".

### What happened (fixed 2026-09-17)

`PUT /api/admin/config` rewrites the whole `config.toml`, so whatever it builds
on decides what survives. It built on `state.shared_state.cfg()` — the live
in-memory config — and never re-read the file.

Editing `config.toml` by hand while the daemon runs is the documented way to set
what the dashboard does not expose; `bootstrap_peers` is the usual one. The live
config does not learn about that edit until a
`POST /api/admin/config/reload`. So an operator who edited the file and then
changed any setting in the dashboard had the whole document rewritten from a
snapshot taken before their edit — the hand-written settings silently gone,
under a toast saying "Settings saved".

### Why this sat open, and why the reason was wrong

It was ranked as "a design question rather than a patch", on the grounds that a
correct read-modify-write "has to be reconciled with runtime changes made
through other endpoints".

**There are no other endpoints.** `SharedState::apply_live_config` has exactly
one production caller besides `reload_config`, and it is this same handler; the
remaining call sites are tests. So no runtime state lives outside the document,
the only way file and live config can diverge is a hand edit, and re-reading is
therefore strictly correct rather than a trade-off.

**The general lesson is the expensive part** (gotcha #631): an entry's stated
reason for being deferred is a claim to check, not a fact to inherit —
especially when the claim is "there are other callers", which is a grep.

### What a change must keep

- **A file that does not parse must NOT refuse the save.** The operator may be
  mid-edit, or it was already broken; losing the change they just made in the
  dashboard helps nobody. `base_for_partial_update` falls back to the live
  config and logs loudly enough to explain why a hand edit did not survive.
- **A second production writer of the live config breaks the argument above.**
  Adding one means this rule needs revisiting, not extending.
- **Consequence, and it is intended**: a hand edit now takes effect on the next
  dashboard save rather than waiting for a reload. Taking effect is strictly
  better than being destroyed.
- The staging-and-rename half (a bare `std::fs::write` truncates first, and a
  crash mid-save left a file the daemon refuses to start on) was fixed earlier
  and is separate; keep both.

### The exception: how the process was started (2026-09-24, FUTURE_WORK #107)

"No runtime state lives outside the document" was false in one respect: the
command line. Config priority is CLI > env > file, and `run.rs` applies the
flags to the config before `SharedState` exists — so both the boot snapshot and
the first live config carry them, and the first save, rebuilt from a file that
cannot hold them, dropped them. Two were exposed, because they touch settings
the dashboard can switch: `--no-update-check` (which had done nothing at all
since v0.3.191, for a separate reason — gotcha #699) and `--anchor`, whose
`apply_anchor_mode` forces auto-manage and `contribution_auto` off. The
environment overrides were audited and all target startup-only settings.

**What changed**: `SharedState::apply_live_config` — the one writer — re-applies
both from the BOOT snapshot, which is the right source precisely because these
are facts about how the process started. Tests:
`a_settings_save_does_not_undo_the_no_update_check_flag` and
`a_settings_save_does_not_turn_an_anchor_back_into_a_model_host`, each red with
its re-apply removed. **A new command-line override of a live setting must be
added to `apply_live_config`**, or it lasts until the first click in Settings.

**From the rules file (moved 2026-10-02):**

`PUT /api/admin/config` rewrites the whole document from a base, so the base
decides what survives. `api::admin::base_for_partial_update` is that choice: the
parsed `config.toml` when it parses, the live config when it is missing or
broken — never a refusal, because a file someone is mid-edit must not cost them
the change they just made in the dashboard.

Building it from `cfg()` destroyed any hand edit made while the daemon ran, which
is the documented way to set what the dashboard does not expose
(`bootstrap_peers`). Re-reading is correct rather than a trade-off because
**`apply_live_config` has exactly one production caller besides `reload_config`,
and it is this handler** — so no runtime state lives outside the document, and
the only way file and live config diverge is an edit worth keeping. A second
production writer would break that argument: add one and this rule needs
revisiting, not just extending.

**The one exception is how the PROCESS was started**, which no file holds:
command-line flags outrank the file, and a save rebuilt from the file drops
them — it would undo `--no-update-check` and turn `--anchor`'s forced-off
auto-manage back on (FUTURE_WORK #107). `apply_live_config`
re-applies both from the boot snapshot; **a new command-line override of a live
setting must be added there**. Every environment override targets a
startup-only setting today, so none is needed for them.

→ `docs/invariants/state-and-config.md`

## config.toml has ONE reader, and a setting's meaningful 0 is never floored

**Rule:** `.claude/rules/arch-state-and-config.md` § "config.toml has ONE reader,
and a setting's meaningful 0 is never floored".

### What happened (report #006, fixed 2026-10-08, FUTURE_WORK #242)

`PUT /api/admin/config` clamped `max_bandwidth_mbps` to `1..=100_000` since an
April sweep added the bounds. In August `0` became AUTOMATIC (10 / 50 Mbps / no
cap by contribution level) — and the floor turned every choice of it into a
1 Mbps cap on serving model parts. The Settings slider labels 0 "Unlimited",
steps by 10 and snaps a stored 1 back to 0, so the panel kept saying
"Unlimited" over a node seeding at 125 KB/s; the request answered `ok`. Before
v0.3.180 every save sent every field, so saving ANY setting did it. A user found
the cap "already set to 1" on 2026-09-11 and was told the setting covers only
model-part serving, which was true and missed this. The same floor hit
`auto_manage_max_storage_mb` (0 = a share of the disk) and `batch_timeout_ms`
(0 = at once).

### Why the repair needed a second fix

`migrate_superseded_defaults` exists for exactly this, a value the daemon wrote
that no person chose. But only the LOADER applied it. A dashboard save
(`base_for_partial_update`), a reload and `GET /api/admin/config` each parsed the
file with a bare `toml::from_str`, so the first save after a start put the
stranded value back into the live config and wrote it to disk again. Every
migration ever added had been undone by the next save. `parse_config_file` is
now the one reader. Its log lines fire once per process, because the settings
read re-parses on every poll.

### What a change must keep

- **A floor belongs only on a setting with no meaning at 0** (`max_concurrent_requests`,
  `max_batch_size`). Before clamping a field, read its doc comment for "0 =".
- **A value a defect stored is repaired only where no UI could produce it.**
  The slider cannot make 1, and 1 MB holds no model part. A deliberate 1 Mbps
  cap is lost, so the smallest honoured cap is 2. That trade is written in the
  book.
- `unknown_config_keys` is the one other parse, and it only round-trips the
  document to learn its schema.

## A platform predicate answers "what kernel is this", not "where am I running"

(2026-09-17, field report #003 against v0.3.182, gotcha #640.)

`config::network::is_wsl2()` reads `/proc/version` for "microsoft"/"wsl". A
container started by Docker Desktop's WSL2 backend inherits the host kernel's
version string **without inheriting the host's networking**, so the predicate is
true for an ordinary Linux container on Windows.

The mirrored-mode probe cannot rescue it either: inside a container namespace
there is no `wslinfo` binary and the interface layout is the container's own, so
`wsl_networking_is_mirrored()` always answers false there. Every containerised
node on Windows therefore took the pessimistic branch and had `listen_address`
forced to `127.0.0.1`, with QUIC, AutoNAT, DCUtR, UPnP and mDNS off.

**Why loopback is the damaging part.** Docker's `-p host:container` mapping
forwards to a listener on a non-loopback interface *inside* the container. Bound
to `127.0.0.1` the peer-to-peer port is published and unreachable. The node
dials out fine and looks healthy from the inside; what it loses is precisely the
NAT'd peers that needed the disabled features. The reporter's only symptom was
"my Docker node sees fewer peers than my Mac on the same account".

**What a change must keep:**

- **The decision stays a truth table.** `wsl_network_adaptation(is_wsl2,
  in_container, mirrored)` returns `None` / `Mirrored` / `NatSafeDefaults` and is
  asserted as one; the previous form asked three questions at the use site and a
  fourth condition had nowhere to go.
- **Detection strictness is set by the asymmetry.** A missed container leaves
  this bug; a container falsely detected on a real WSL2 shell undoes gotcha
  #161's mirrored-mode fix. So only signals a bare WSL2 shell cannot produce
  count — verified on this project's own WSL2 box: `/.dockerenv` absent,
  `/run/.containerenv` absent, `container` unset, `/proc/1/cgroup` =
  `0::/init.scope`.
- **Several signals, OR'd.** No single one survives every runtime and cgroup
  version; cgroup v2 inside a container can read a bare `0::/`, so the cgroup
  path cannot be the only signal.
- **`health::monitor::maybe_warn_wsl_firewall` is correct today by accident.**
  It gates Windows-firewall advice on `is_wsl2() && wsl_networking_is_mirrored()`,
  and the mirrored probe already fails in a container. If that probe ever learns
  another signal, that site needs the container exclusion too.

**Generalisable**: a predicate named for a PLATFORM answers "what kernel is
this", not "what environment am I in". Ask what else inherits the signal before
letting it choose settings.

**From the rules file (moved 2026-10-02):**

**`config::network::wsl_network_adaptation(is_wsl2, in_container, mirrored)` is the one
decision** about WSL2 network overrides, returning
`None` / `Mirrored` / `NatSafeDefaults` as a truth table rather than three
separate calls at the use site.

`is_wsl2()` reads `/proc/version`, so it is **true inside a Docker container on
Windows**, which inherits the host kernel's string and none of its networking.
That forced `listen_address = 127.0.0.1` on every containerised node — a
published container port cannot reach a loopback listener, so the node was
unreachable while looking healthy (report #003, gotcha #640). **This is the
SECOND time this predicate proved too broad**; gotcha #161 was the first.

`running_in_container()` ORs several signals because none survives every runtime
and cgroup version. Keep it strict: a missed container leaves the bug, but a
false positive on a real WSL2 shell undoes #161's fix. Before letting any
platform predicate choose settings, ask what else inherits its signal — a
container, a VM, a chroot, an emulator.

→ `docs/invariants/state-and-config.md`

## Single-source-of-truth helpers — SharedState, live config and credits

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/state-and-config.md`

- **`SharedState::release_request_state`** — clears the maps a finished request leaves behind: `active_pipelines`, `active_traces`, `request_holder_blacklist`, `peer_vram_commitments`, `local_memory_refusals`, `route_plan_overrides`, `salvaged_replies` and `retained_activations` — keyed by request id, sharing one lifetime. Deliberately does NOT touch `active_count` or `queue_notify`.
- **Credits are DORMANT — nothing may publish or act on a balance** — `MIN_BALANCE_FOR_INFERENCE = 0` and `calculate_tier` returns `DORMANT_TIER` whatever it is given, so no balance affects who is served or how fast, and the leaderboard neither ranks by credits nor publishes them.
- **`SharedState::cfg()`** — the live config, and the single answer to "what is this setting **now**".
- **`SharedState::record_peer_serve`** — the single answer to "this node did inference work for a peer", counting it AND billing for it.
- **`config::InferenceConfig::claims_shard`** — the single answer to "does this node claim shard N?", i.e. how `inference.shard_range` is read. **Never read `shard_range` directly.**
- **`SharedState::local_executor_serves(&request)` is the single answer to "may the singleton executor (a whole GGUF) answer this request?"** — it holds THIS model, and the request names no LoRA adapter (it applies none). It takes the REQUEST: as `(model_id)` it could not see the adapter, and three gates (`execute_batch` — every request, not the first — `PipelineExecutor::execute`, `execute_request`) answered LoRA requests from the base model, the last one any model from the resident one (gotcha #711). Guard: `the_singleton_executor_is_handed_a_request_only_through_one_predicate`.
- **`SharedState::local_fast_path_for` is the single answer to "may this request take the local split fast path?"** — both API surfaces used to compose it themselves (`has_complete_split_model && !should_offer_work_to_the_swarm`), and the fast path skips the router — which is where `delegation_target` lives. **It takes the request's `swarm_route` override as a REQUIRED argument**, because it decides whether the request ever reaches the router and the router is the only reader of that override: without it the knob was inert on every node holding a whole model, which auto-manage deliberately converges nodes on being (gotcha #633). The two dispatch callers pass what the caller asked for; the admin LISTING passes `None`, because a listing answers for the node and not for a request. `override_keeps_whole_model` is the pure decision, a truth table beside `local_fast_path_allowed`. **This fast path has now eaten a feature three times** (#187, #443, #633) — when adding anything request-scoped, grep for every early return between the API edge and the code that reads it.

## SharedState fields — the full per-field notes (moved 2026-10-02)

- `state.models.foreign_wishlist` — R130. `DashMap<(NodeId, ModelId), (score_0_100, received_at_ms)>`; capped at `MAX_FOREIGN_WISHLIST_ENTRIES = 10_000` with oldest-first eviction, 2h freshness window enforced on read. Written by `apply_wishlist_announcement` on inbound `SwarmMessage::WishlistAnnouncement`; read by `compute_wishlist` for the 0..10 cross-pool demand boost.
- `state.models.quant_recommendations` — R133. `ArcSwap<QuantRecommendations>`; refreshed via `crate::model::auto_manage::quant::refresh_quant_recommendations(state)` on every auto-manage tick AND on every WS stats build. Read by `GET /api/admin/quant-recommendations` and the swarm-tab tips tile.
- `state.models.shard_download_claims` — the shards a download task is WRITING right now, one RAII `ShardDownloadClaim` each, taken by `claim_shard_download` and released only by dropping it. `is_shard_in_progress` reads it; `shard_marked_in_progress` is the map-only sibling for a caller that already holds the claim. **Exclusion between writers must not rest on `acquisition_progress`** — that is a progress structure with several writers and a timer-driven deleter, and coupling the guard to it has failed in the field twice. → `docs/invariants/network.md`
- `state.models.shard_download_backoff` — external report 2026-07-23. `DashMap<ShardId, ShardDownloadBackoff { fail_count, retry_after: Instant }>`. Exponential per-shard download cooldown (30→60→120→240→300s cap, via the pure `shard_backoff_delay_secs`). Recorded via `record_shard_download_failure` at every terminal *transient* download-failure site (HF `download_shard` error + GGUF-probe failure in `model/auto_manage/download.rs`, P2P give-up-with-no-HF-source in `network/manager/shard_transfer.rs`, and stall-reconciliation in `health/monitor.rs::cleanup_acquisition_progress`). Checked by `shard_in_backoff` in `scoring.rs::gather_candidates` (skips the shard while cooling down). Cleared via `clear_shard_download_backoff` on success (HF success arm + P2P completion in `requests.rs`). Distinct from `shard_p2p_failed`, which only *forces* the HF path without throttling re-selection — the two solve different problems and a new failure site should touch whichever it needs. Do NOT record backoff on the P2P→HF fallback branch: that path wants an *immediate* HF retry. Entries self-evict from `shard_in_backoff` once idle past `SHARD_BACKOFF_FORGET_SECS` (1h), so the map stays bounded without a dedicated sweep.
- `state.models.manifest_heard` — 2026-09-23, re-keyed 2026-09-25. `DashMap<(ModelId, [u8; 32]), Instant>`: every manifest VERSION the WHOLE SWARM was gossiped in the window, and when. **Per version, not per model** — holding only the last hash per model made holders that disagree about a part (#61) each forget their own version the moment the other's arrived, so both re-announced every round (~90% of manifest gossip, measured 2026-09-25). Written only by `note_manifest_heard` — from the dispatcher's `ModelManifest` arm AFTER `verify_hash_strict`, and from `HealthMonitor::broadcast_manifests` for our own broadcast. **It takes the `MessageTransport` as a required argument and ignores `Direct`**: a point-to-point catch-up reached this node alone, and counting it would let reconnects keep every holder of a model quiet. Read only by `manifest_heard_within` (exact hash, inside a window), which is Trickle's suppression test (RFC 6206); swept each broadcast round by `forget_manifests_heard_before`, so it holds at most one window's worth. → `docs/invariants/network.md` § "The repetition, not the size"
- `state.models.hf_sources` / `origin_claims` / `canonical_builds` / `origin_refusals` / `canonical_holding` — 2026-10-02 (#151, v0.3.221). One upload per model id (`model::canonical`). `origin_claims` = every upload heard of, best-first, bounded; `canonical_builds` = the verified choice (persisted, tree `canonical_builds`, restored in `SharedState::new` BEFORE anything reads `hf_sources`, which it overrides); `origin_refusals` = uploads HuggingFace would not serve, with a retry time; `canonical_holding` = what this node's copy is against the choice (`Holding`), read by the listing and the acquisition gate. ⛔ **`hf_sources` is written ONLY by `SharedState::write_hf_source`** (behind `note_origin_claim` / `adopt_canonical_build`) and the startup restore — guard `a_models_source_is_written_only_through_the_canonical_choice`; a header is fetched only by `fetch_model_header` — guard `a_models_header_is_fetched_only_from_the_upload_its_parts_are`. Every download path asks `canonical_allows_acquisition`. → `docs/invariants/network.md` § "One upload per model id"
- `state.metrics.peer_outliers` — 2026-10-02. `PeerOutliers`: per-(peer, model) consecutive failures and ejection (Envoy-style). Fed ONLY by `record_peer_delivery` (model a required `Option` argument) and reset by `note_peer_completed_request`; read by the scheduler's `gather_candidates`, which still admits an ejected peer for a part nobody else holds. → `docs/invariants/scheduling.md` § "A peer that fails a model on every request"
- `state.models.auto_model` — 2026-09-26 (#120). `parking_lot::Mutex<Option<ModelId>>`: the model `auto` last resolved to. Read and written ONLY by `api::openai::resolver::auto_model_for`, which keeps answering with it while it is still servable, so a conversation on `auto` stays on one model. **Never make `auto` read `loaded_model_info` again** — the shard scan overwrites it with every model it registers, partial holdings included.
- `state.models.removed_by_user` — 2026-08-21 (gotcha #360). `DashMap<ShardId, bool>`, persisted in DB tree `removed_shards`, loaded in `SharedState::new` like `locked_shards`. A shard the USER deleted (`delete_shard`, `delete_model` — every manifest shard) is an instruction, not a gap: `gather_candidates` skips it unless `in_configured_range || pinned_to_us`; an explicit request clears it (`hf_download_shards` for the named shards, `download_shard`, `pool_add_pin` naming this node). Helpers live in `daemon/state/removed_shards.rs` (`mark_shard_removed_by_user`, `shard_removed_by_user`, `clear_shard_removed_by_user`, `clear_removed_by_user_for_model`); the shard listing emits `removed_by_user` (only when not local) and the dashboard shows a "Removed" badge. Never write the map or the tree directly.
- `state.models.disputed_shards` — shards this node HOLDS whose bytes disagree with the swarm's hash and which are KEPT and served anyway, because that hash has no origin backing (`ModelRegistry::mismatch_policy`). Written and cleared only by `SharedState::note_shard_disputed` / `clear_shard_dispute`, from the **three** paths that compute `mismatch_policy` and can land on `KeepBytes`: the startup verification sweep, the auto-manage rescan, and the pending-verification drain (`auto_manage::manager::verify_pending_shards` — silent until 2026-09-14, and the path a PEER-PROVISIONED node actually reaches, so the report read `0` while the warning fired twice). All three clear on a later successful verify, which is the only thing that settles a dispute, and all three clear on EVERY success rather than only where a dispute is known — a clear that has to be predicted is a clear that gets forgotten. Passing `KeepBytes` as a CONSTANT to ask whether a file is already on disk is a question, not an acceptance, and records nothing. Guard: `every_path_that_keeps_disagreeing_bytes_records_the_dispute`. **Deliberately NOT `shards_needing_repair`** — that set's drain clears any mark whose file is on disk, which is every shard on this path. It exists so the disagreement is countable from OUTSIDE the log: the diagnostics report prints the section even at zero, because a pasted report saying `0` is a measurement and one that says nothing is not. **Read it through `SharedState::disputed_shards_now`, which self-evicts entries whose file has gone** — a dispute also ends when the shard is deleted or pruned, and those three call sites do not know about the set; a phantom would corrupt the very figure the set exists to measure. → `docs/invariants/network.md`
- `state.credits.foreign_pool_catalog` — R134. `DashMap<(PoolId, ModelId), received_at_ms>`; capped at 5000 with oldest-first eviction, 2h freshness window. Written by inbound `SwarmMessage::PoolModelAvailability` handler. Read by `GET /api/admin/foreign-pool-catalog` and by `pool::scope::cross_pool_extras` (R134.7) when `pool.allow_cross_pool_inference` AND `private_mode` are both on.
- `state.local_memory_refusals` — 2026-09-07. `DashSet<Uuid>` on the ROOT
  SharedState, beside `request_holder_blacklist` and released by
  `release_request_state` with it. Written ONLY by
  `note_local_memory_refusal`, read only by
  `local_memory_refused_for_request`. It is the re-plan's queueing hint — see
  "A re-plan is warranted by a changed fact, never by a failed attempt".
- `state.encrypted_pipeline_models` — **never read directly.**
  `SharedState::encrypted_pipeline_for` is the single answer to "is prompt
  privacy on for this model"; the map holds only the EXPLICIT per-model toggle
  and misses both automatic cases, one of which
  (`encrypted_pipeline_auto`, default ON wherever the node holds both ends) is
  how privacy is normally switched on at all. Use
  `privacy_explicitly_enabled_for` only where a DELIBERATE user choice is the
  question. `prompt_privacy_is_never_re_derived_from_the_per_model_map` in
  `tests/repo_consistency.rs` fails the build on a direct read.
- `state.region_demand` / `state.local_region_demand` — 2026-09-21. TWO maps, and the split IS the fix. `region_demand` is the MERGED view `(model, region) → rate`, written by the inbound `ModelDemandGossip` handler as well as locally, and read by auto-manage scoring, pruning and the wishlist, which all want the whole picture. `local_region_demand` is `model → rate` for THIS node's own region — the EMA of `models.model_request_counts`, maintained by `auto_manage::manager::decay_request_counts`, whose `old` value comes from this map so a same-region peer's gossip cannot inflate the figure we then publish as our own measurement. ⚠ **Only the local map may be GOSSIPED.** Publishing the merged one re-originated every peer's demand under this node's id every 30 s, refreshing the timestamps the staleness check relies on so entries could never age out — a node that had served zero requests published ~93 demand messages per tick about other people's traffic. The two have different key types, so the wrong one no longer compiles at the publish site, and `the_demand_we_gossip_is_the_demand_we_measured` guards the rest of the file. **GossipSub already propagates the originator's message; re-originating is not what makes it travel.** → `docs/invariants/network.md` § "Gossip volume"
- `state.metrics.gossip`'s per-topic MESSAGE counts (`GossipTopicTotals`; this entry named a field `inference_counts` until 2026-10-02, which never existed) — the per-topic MESSAGE counts (`published`, `sent`, `recv`, `recv_unfiltered`) are parsed beside the byte counters, because bytes alone cannot say whether a topic is expensive from size, frequency or duplication. `published` vs `sent` separates speaking from relaying; `recv_unfiltered` vs `recv` is the duplicate factor. ⚠ **`sent` counts ATTEMPTS, once per recipient** — `msg_sent` runs before the queue that can refuse the message — so a topic's `sent_bytes` may legitimately exceed `BandwidthMeter`'s total and **`sum(parts) <= whole` must not be asserted**. The three `*_messages_dropped_per_topic` families (queue expiry) and `GossipMeter::send_failures` (queue full, from `gossipsub::Event::SlowPeer`, which no metric covers) are what make the gap readable. Gotcha #674.
- `state.metrics.bandwidth` — 2026-09-12. `Arc<BandwidthMeter>`; libp2p's transport counters, armed once when the swarm is built (`with_bandwidth_metrics`, the ONLY builder phase where a transport can be wrapped) and read back by encoding the registry. **`totals()` answers `None`, never `0`, when nothing is counting** — a figure that reads zero whether the node is silent or the counters were never wired is the reading this replaces. `refresh()` is called from the health-monitor tick and NOWHERE else, because a rate needs two readings at a known cadence; everything else reads `current()`. libp2p's transport counters, armed once when the swarm is built (`with_bandwidth_metrics`, the ONLY builder phase where a transport can be wrapped) and read back by encoding the registry. **`totals()` answers `None`, never `0`, when nothing is counting** — a figure that reads zero whether the node is silent or the counters were never wired is the reading this replaces. `refresh()` is called from the health-monitor tick and NOWHERE else, because a rate needs two readings at a known cadence; everything else reads `current()`.
- `state.metrics.gossip` — 2026-09-21. `Arc<GossipMeter>`; GossipSub's own per-topic byte counters, armed in `build_behaviour` (the ONE point `with_metrics` can attach, since it consumes and returns the behaviour). **Its own registry, not `bandwidth`'s** — the swarm builder holds a `&mut` borrow of that one while it runs, and the two are independently absent: the transport counters appear on the first byte of any protocol, these on the first gossip message SENT TO A PEER. An isolated node therefore reads `None` for ever and that is correct, not broken (`msg_sent` is recorded per recipient inside `send_message`). **Answers `None`, never `0`** — same reason as `bandwidth`. **Warns about nothing**, deliberately: every node reads `None` for its first seconds because `prometheus_client` writes a `Family`'s rows only once a label set exists, and a diagnostic on that fires on every start (gotcha #582). ⚠ **`libp2p-gossipsub` is a DIRECT dependency solely to enable its `metrics` feature** — the `libp2p` facade's `metrics` feature does not — and if that requirement ever drifts from libp2p's pin, cargo builds two copies and the split reads as silently absent. Guard: `the_gossip_counters_come_from_the_same_crate_libp2p_uses`.
- `state.metrics.inference` — 2026-09-21. `Arc<InferenceTraffic>`; the inference half of the traffic split, **counted in the CODEC** (`network/protocol/mod.rs`) and nowhere else. ⚠ **Not at the send sites, and that is the whole point**: `network.tensor_compression` defaults ON, so `dispatch_tensor_payload` holds the UNCOMPRESSED activation and a counter there reports bytes the interface never carried. `other_*` is the total minus the named categories, so an over-count corrupts the remainder as well as itself. `counts_as_inference` excludes shard transfers (counted at the cap's choke point — twice would break the sum) and `RelayedTensor` (somebody else's work, already `relay_bytes_forwarded`), and INCLUDES `StreamingToken`, which is most of what a whole-model node's inference costs. Counted as the full frame, header included, in both directions. Plain atomics, so zero genuinely means zero — unlike `bandwidth` and `gossip`, nothing here can be "not counting yet".
- `state.metrics.last_dispatch_at_ms` + `last_dispatch_kind` — 2026-09-19. Message-dispatcher liveness: epoch millis of the last message taken off `network_out`, and that message's `SwarmMessage` variant name. **Written by the dispatcher (`metrics.note_dispatch`, immediately after `recv()` returns and BEFORE the `match`, so it covers every arm including the `continue`s), read by `HealthMonitor::report_dispatcher_stall` on its own tick.** The split across two tasks is the point: `daemon::supervisor` reacts only when `JoinSet::join_next()` returns, i.e. to a panic or a clean exit, so a task parked for ever inside an `.await` produces no signal — 45 minutes of total silence produced not one supervisor line (`docs/FUTURE_WORK.md` #90). A heartbeat emitted from inside the dispatch loop would be just as silent, for the same reason the loop is stuck. `0` means nothing has been dispatched yet, which is not a stall. **A stall is a message WAITING, not a dispatcher idle** — `dispatch_stalled_for` (read by both the report and `inference_outage`) reads `network_out`'s depth through `metrics.dispatch_queue` (a `WeakSender`, set once in `daemon/mod.rs`) and times it from the later of the last take and the last time the queue was seen empty; idle time alone fired for 52 minutes on a node whose host had lost its network (2026-09-25). Guard: `the_dispatcher_liveness_marker_is_written_before_the_match_and_watched_elsewhere`. **Past `DISPATCH_STALL_AFTER` the node also WITHDRAWS inference from what it advertises**, through `SharedState::inference_outage` — the one predicate it shares with the dead-graphics-stack case, so two unrelated outages cannot advertise different things about the same node. Shard serving is untouched; `NodeCapability::can_serve_inference` carries it and defaults to `true` on the wire. Guard: `both_inference_outages_withdraw_through_one_predicate`.
- `state.metrics.swarm_capacity` — R110. ArcSwap<SwarmCapacity>; refresh via `crate::daemon::state::refresh_swarm_capacity(state)`. Eagerly refreshed on peer connect (`network/manager/identify.rs`) and disconnect (`network/manager/connections.rs`) so the dashboard banner stays consistent with the peer-list panel under churn — the WS stats-cache 1.5s coalesce alone is too lazy.
- `state.metrics.segment_latency` — `Arc<SegmentLatencyTracker>`: per-(model, segment, holder) forward-latency EWMA and sample count, written by `SharedState::record_segment_latency` from the distributed forward-success path and read by `peer_performance_rows` (the per-peer performance table); `evict_stale` on the HealthMonitor tick bounds it. It was the measuring half of hedged verify dispatch, **removed 2026-09-24 (FUTURE_WORK #94)**: a hedge sent a decode-time verify to a holder with no cache for that conversation, so it could never be right, and making it right costs a prompt-pass replay per hedge. Do not re-add hedging of a stateful step; failover with `assemble_replay` is the mechanism.
- `state.metrics.prefetch_orchestrator` — R136 Layer 3. `PrefetchHandle` (Arc<PrefetchOrchestrator>) with per-session first-token histogram + idle-time learner + throttling. Observation via `observe_user_turn(session, first_token)` + `record_response_completion(session, now_ms)` at the router success site. R142.6 wired `evict_idle` to the HealthMonitor tick to bound the histories map. K-layer prefetch dispatch is the remaining integration; data-collection and orchestration are complete.
- `state.standalone_tokenizers` — R136 Layer 1/3 follow-on. `DashMap<ModelId, Arc<SplitTokenizer>>` on the ROOT SharedState (not a sub-struct — used by both `state.metrics`-derived L3 prefetch AND the `pipeline/ngram_only_spec.rs` L1 path, so cross-cutting). Lazy-loaded from `gguf_header.bin` via `state.standalone_tokenizer(&model_id)` accessor. Returns `None` when the header isn't on disk; caller falls through gracefully.
- `state.pending_activation_chunks` — R139 Tier 4K. `DashMap<Uuid, ChunkAssemblyState>` on the ROOT SharedState (cross-cuts the RR-decrypt path in `network/manager/tensors.rs` and the persistent-stream reader in `network/pipeline_stream.rs`). Receiver-side assembly for STREAM-chunked activation forwards. Entry-locked insert via `state.try_assemble_chunked_forward(forward, sender_peer_bytes)`. Periodic stale-entry sweep wired to the HealthMonitor tick via `state.sweep_stale_chunk_assemblies(ttl_secs)`. Chunk-meta is bound into AAD via `build_layer_forward_aad`, so reorder/truncation/cross-transfer-substitution fail Poly1305 before reaching the assembly.
- `state.listen_multiaddrs` — R140. `arc_swap::ArcSwap<Vec<String>>` on the ROOT SharedState (cross-cuts NetworkManager-writes and PoolManager-reads). Live snapshot of the swarm's reachable addresses, each terminated with `/p2p/<local_peer_id>`. Written by `NetworkManager::refresh_listen_multiaddrs()` (events.rs) on `NewListenAddr` / `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed` / UPnP `NewExternalAddr` / `ExpiredExternalAddr`, plus once at startup after `listen_on()` (and after the `network.external_addresses` config override is added). **R143: the snapshot is the UNION of `swarm.listeners()` (bound sockets — private LAN on a NAT'd node) AND `swarm.external_addresses()` (UPnP-mapped / AutoNAT-confirmed / relay-circuit / manually-declared public addrs).** Without the union a NAT'd node's invite code silently shipped a LAN-only address. Built via the extracted, unit-tested `build_reachable_multiaddr_list(candidates, peer_id)` + `ensure_p2p_suffix` helpers; filtered through `addr_is_remotely_reachable` — keeps LAN + Tailscale CGN (100.64.0.0/10) + public, drops loopback / unspecified / link-local / IMDS. Read by `PoolManager::handle_generate_invite_code` when minting v2 `swarmpool://` codes; empty list → `SwarmError::ServiceUnavailable`. When the list has entries but NONE pass the stricter `pool::invite::any_internet_reachable` (public IP / DNS / relay-circuit — excludes LAN + CGN), invite generation still succeeds but emits a `pool`/`invite_lan_only` warning ActivityEvent so the user isn't handed a LAN-only code that dies over the internet.
- `config.api.dashboard_trust_lan` — read via `SharedState::cfg()` (see below), never re-derived with `addr.ip().is_loopback()`. `api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"; the sibling `dashboard_trust_overlay` is read the same way. Was a private `AtomicBool` mirror until 2026-08-09, folded into the live config when that became general.
- **`api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"** Do NOT re-derive it with `addr.ip().is_loopback()`. That predicate means "the last TCP hop began inside this daemon's network namespace", which is simultaneously broader than intended (a same-host reverse proxy such as `tailscale serve` satisfies it on behalf of a fully remote client) and narrower (a container publish, a NAT, or a Tailscale subnet router never satisfies it — not even from the host's own `localhost` — because subnet routers SNAT by default). Same-origin checks belong on `Origin` vs the request's own `Host` (`websocket.rs::ws_origin_allowed`), never on a hardcoded loopback allowlist: that mistake independently cost every non-loopback dashboard its live WebSocket updates. See gotcha #195.
- `state.relay_proven_features` — `DashMap<NodeId, RelayProvenFeatures { features: u64, proven_at: Instant }>` on the ROOT SharedState (`daemon/state/relay.rs`). Records relay features a peer has *demonstrably* used by relaying a message addressed to us: `handle_relayed_tensor` records `features::TENSOR_RELAY`, `handle_relayed_envelope` records `features::RELAY` (via `record_relay_proven_features`, which ORs bits + refreshes `proven_at`). The relay send path's feature gates (`target_supports_{relay,tensor_relay}` in `network/manager/relay.rs`) consult `relay_feature_proven(peer, bit)` FIRST, before the gossiped `NodeCapability.features`. **This is the cold-start return-path fix**: a serving node reaches a coordinator known only via `ensure_relayed_origin_known` (whose `peer_registry` entry has `capability: None`, because the capability-gossip handler at `daemon/dispatch/mod.rs` is update-only and can't populate a not-yet-existing entry). Without the proof, the return relay of a computed `LayerResult` was refused until a capability-gossip round landed (≤30s), dropping the first result. Freshness = `RELAY_ROUTE_TTL_SECS` (re-proven on every inbound relayed message, so an active session never goes stale); swept alongside `relay_routes` in `sweep_stale_relay_state`. New relay send paths that gate on a peer's relay capability MUST consult this proof, not just the gossiped capability.

(The rules file repeated the sentence below twice; the second copy is kept here only so no word is lost.)

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.
