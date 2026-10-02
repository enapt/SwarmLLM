# Network protocol, peers and the model registry

The evidence behind the rules in `.claude/rules/arch-network.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## A disconnect retires a session key; it must not destroy it

`SessionManager::remove_session` moves the live key into `retired` — openable,
never sealable, for `PREVIOUS_KEY_GRACE`, carrying its own replay window — and
`open` falls back to it after the current and superseded keys, including when
there is no session at all.

**Why.** The removal exists to force a fresh handshake and stop epoch desync,
and that part is right. Destroying the key with it is not: a forward sealed
moments before the drop then cannot be read, and the `previous` slot that exists
for exactly this problem goes with the entry.

**The two ends do not drop together, and that asymmetry is the bug.**
`handle_connection_closed` keeps the session when the peer is
`in_active_pipeline` — but that reads `active_pipelines`, which is the
COORDINATOR's map and holds nothing for work a node is SERVING for someone else
(gotcha #194). So on a brief drop the server clears its session while the
coordinator keeps sealing with the old key, and every forward in flight fails to
decrypt. The serving side has no equivalent signal to consult: between decode
tokens it has no inbound forward outstanding at all, so "is work in flight" is
false precisely when the request is alive.

Measured on v0.3.164 (report #028): a 4m43s generation, already streaming, died
outright when its tail peer's connection dropped and the retry reached the same
node with `Could not decrypt forward`. The identical signature was recorded five
weeks and ninety-five versions earlier and closed as a rotation race on one
peer; it is the same shape seen from the other side — a key one end threw away.

Four things a change here must keep.

- **Retired keys OPEN, never SEAL.** That is what keeps this from reintroducing
  the nonce reuse the removal exists to prevent, and it is pinned by
  `a_retired_key_cannot_be_used_to_seal`.
- **The reconnect still handshakes afresh.** `retired` is a separate map, so it
  does not satisfy `establish_session`'s idempotence guard.
- **Its own replay window travels with it**, so this is a second authenticated
  check rather than a relaxed one — WireGuard's per-keypair counter, the same
  detail that made the previous-key grace safe when it was added.
- **The window is bounded and swept.** `evict_stale` drops retired keys past the
  grace period; holding one longer widens the window in which an old key opens
  anything, for no benefit.

**A test here must use an EPHEMERAL session.** `establish_session` derives from
long-term identity keys, so a reconnect re-derives the identical key and a
static-key test passes with the fix reverted — which is how the first version of
`a_forward_in_flight_survives_the_peer_reconnecting` was written, and a null
control caught it. Forward secrecy means the real link is ephemeral and a
reconnect genuinely changes the key.

### The mirror image: a key that must not be KEPT (2026-09-12, report #016)

Everything above is about a key destroyed too eagerly. The same asymmetry has a
second failure, in the other direction, and the paragraph above describes the
mechanism without drawing the conclusion: **the `in_active_pipeline` exemption
was also wrong for the side that KEPT its session.**

The coordinator kept an ephemeral key across a reconnect. The serving node — not
in `active_pipelines`, because that map is the coordinator's — retired its own
and came back on a fresh static one. Nothing then noticed: `seal` succeeds
whatever the peer holds, the failure happens on the far side, and
`establish_session` is idempotent, so the Identify that follows the reconnect
left the stale key exactly where it was. 29 forwards to that peer, 29 `Could not
decrypt forward`, zero successes, every request routed through it dead until one
end aged the session out ten minutes later.

The exemption dated from April 2026, when `establish_session` reinstalled on
every Identify and its comment — "reconnection will refresh it" — was true. It
stopped being true when that became idempotent, and **the comment is how the
contradiction survived**: a claim about another function's behaviour goes stale
silently, which is why `a_disconnect_retires_the_session_even_mid_pipeline` in
`tests/repo_consistency.rs` is a test and not a comment.

What it was protecting is gone too. It bought a sealable key across the gap, but
a peer we are not connected to cannot be sent to, and after the reconnect that
peer has no session to open with — so the seal it saved produced a forward
nobody could read.

**And a session the other end cannot open now repairs itself.** A failed `open`
arms a repair (`SessionManager::request_rekey`, at that single choke point so a
third decrypt site inherits it) and `crypto::key_rotation` performs one ephemeral
exchange, which both ends install. That covers every other way the two can
diverge — a lost exchange reply, two rotations crossing — none of which either
end can detect locally. Rate-limited per peer, because a broken session fails
every forward of every request and one exchange repairs all of them: without the
limit, 29 failures would have meant 29 handshakes.

A forged or replayed frame also arms a repair. That is deliberate and costs
nothing — connections are authenticated by PeerId at the Noise layer, so only
that peer can trigger it, and it can always ask for a fresh key anyway.

### The third failure: a key only ONE end ever had (2026-09-21, field-reported on v0.3.193)

The repair above treats divergence as something to recover from. This is where
it came from, and it was being manufactured on a timer.

**A rekey is two messages, and the second one can be lost.** The responder
derived the new key and installed it *before* answering, so if its answer never
arrived the initiator kept the old key while the responder had moved on. From
that moment nothing the responder sealed could be opened — in ONE direction,
which is why neither end could see it: the initiator's forwards still arrived
and were still answered.

**Measured, not reasoned about.** A peer's forward reached this node under a key
it had never installed — `recv_nonce=0`, the first message of a session, 102 s
after this node's own rotation tick — and the request died with `Could not
decrypt forward`. The same peer's own report that evening carried two more, on
two different peers, in two different topologies, both on the FIRST segment of a
pipeline. One sender, three receivers, one signature: the divergence was being
made by whoever answered an exchange, not by any particular peer.

Three ways an answer is lost, and none of them is exotic: the reply is a
fire-and-forget `SendDirectMessage` with no delivery id, so libp2p's
request-response layer may drop it silently under load; `network_tx.try_send`
drops it when the channel is full; and the peer may be unresolvable for the
moment it takes to answer. The comment beside the send said a dropped reply
"re-runs on the next exchange" — true of the reply, false of its consequence,
because the responder had already committed.

**The rule: a key derived while ANSWERING an exchange does not take effect until
the peer proves it has it.** This is WireGuard's, for the same reason — a
responder may not send under a new keypair until it has received one transport
message under it, because only that proves the initiator got the handshake
response ([wireguard.com/protocol](https://www.wireguard.com/protocol/)).

The implementation is `SessionManager`'s `unconfirmed` slot:

- `accept_ephemeral_exchange` parks the derived key there instead of installing
  it, and keeps sealing with the key both ends still agree on. With no session
  at all it installs as before — there is nothing else to seal with, and no
  working state to protect.
- `open` tries the parked key after the live one and, when it opens, promotes
  it. **The window it accumulated travels with it** (`install_with_window`), or
  the message that confirmed it could be replayed under the promoted key.
- The initiator seals `SESSION_CONFIRM_MARKER` the moment it installs;
  `complete_ephemeral_session` RETURNS those bytes so the seal and the install
  cannot come apart, and the dispatch handler sends them as
  `SwarmMessage::SessionKeyConfirm`.
- **Ordinary traffic confirms a key just as well.** The message is a prompt, not
  the proof — which is what keeps an older initiator working: it never sends one
  and is adopted the moment it forwards anything.

**Gated on `features::SESSION_KEY_CONFIRM`, and the gate is not decoration.** A
peer that cannot confirm must be answered the old way, or the failure is simply
mirrored: it would retire the key we kept waiting on and nothing we sealed would
open. Unknown reads as "cannot confirm", which is what an absent capability
gives.

**Answering an exchange also drops our own outstanding initiation.** Two
rotations crossing used to leave each end sealing with a key the other held only
as superseded — fine for `PREVIOUS_KEY_GRACE`, then broken in both directions at
once. Now at most one new key per link per round: if they genuinely crossed,
neither takes and the next tick tries again with nothing broken in between.

**And a peer we hold NO session for now arms a repair too.** That path returned
`NoSession` and armed nothing, on the reasoning that there was no session to
repair — but an exchange needs no session to run, and "no key" is not a milder
version of "the wrong key", it is the same dead link.

Guards: `key_confirmation_tests` in `src/crypto/session.rs`, including
`adopting_a_key_before_the_peer_has_it_breaks_one_direction` — the old behaviour
kept as a running null control, because the fix and the defect differ by one
enum value and a test that cannot tell them apart is not a test.

**What this does NOT fix.** A decrypt failure still ends the request it lands
on. The repair arms in the same call and completes in a round trip, but by then
`failover_segment` has already exhausted a segment that usually has no standby
(every single-peer delegation, by design) — so the user sees `Segment 0 failed
with no standby available`. Making a known-transient, known-self-repairing
failure survivable is separate work: see `docs/FUTURE_WORK.md`.

**From the rules file (moved 2026-10-02):**

`SessionManager::remove_session` moves the live key into `retired` — openable,
never sealable, for `PREVIOUS_KEY_GRACE`, carrying its own replay window — and
`open` falls back to it after the current and superseded keys, including when
there is no session at all.

**It runs on every full disconnect, with no exemption.** An active pipeline is
not a reason to keep a session: `active_pipelines` is the COORDINATOR's map, so
the serving peer retires its own either way and comes back on a different key,
and `establish_session` is idempotent — nothing repairs the mismatch.
`a_disconnect_retires_the_session_even_mid_pipeline` in
`tests/repo_consistency.rs` fails the build on a gated call.

**A session the peer cannot open repairs itself.** A failed `open` calls
`request_rekey` (inside `open`, so every decrypt site inherits it) and
`key_rotation` performs one rate-limited ephemeral exchange — including when
this node holds NO session for that peer, which is the same dead link, not a
milder one. **The forward it refused is sent again once that repair lands**:
the refusal carries `ForwardRefusal::Undecryptable` (the `0x06` result
trailer, gated on `features::FORWARD_REFUSAL_REASON`) and the coordinator's
`wait_for_result` resends on `rekeyed_since` — `arch-scheduling.md`.

→ `docs/invariants/network.md`

## A disagreement nobody can read is not an instrument

(2026-09-13.) **`state.models.disputed_shards` is the record of "we checked,
we disagree, and we are keeping our copy"**, written and cleared only through
`SharedState::note_shard_disputed` / `clear_shard_dispute`.

**Why it exists.** `mismatch_policy` (the rule above) stops a node destroying
its own bytes on a hash the model's origin never backed. It does not settle the
argument, and settling it is open work whose stated precondition is *how often
this fires in the field*. That precondition could not be met by anyone: the
count was a `u32` local to the startup verification task, logged once per
daemon run and dropped. Nothing on the dashboard, nothing in the diagnostics
report a reporter pastes, nothing in the API. The safety net shipped in
v0.3.177 was therefore field-unverifiable by construction.

**What a change must keep.**

- **Not `shards_needing_repair`.** `complete_pending_shard_fetches` begins by
  treating any shard whose file is on disk as already repaired and clearing the
  mark — and a disputed shard is on disk by definition. Marking one fetches
  nothing. The quarantine path only ever worked because it deleted the file
  first, and the first cut of the .177 fix called it anyway and claimed a
  settlement that could not happen.
- **Cleared on every successful verify and every quarantine**, not only where a
  dispute is known to exist. `DashSet::remove` on an absent key is free, and a
  clear that has to be predicted is a clear that gets forgotten. A quarantined
  shard is repaired, not disputed.
- **The diagnostics section prints at zero.** A pasted report saying `0` is a
  measurement; one that says nothing is not, and the whole point is to find out
  whether this ever happens.
- **The read self-evicts.** `SharedState::disputed_shards_now` drops any entry
  whose file has since gone, and both the count and the report go through it. A
  dispute ends three ways — a later check passes, the bytes are quarantined, or
  the file stops being here (the user deletes the part, `delete_model` takes the
  lot, auto-manage prunes it). The first two clear where they happen; the third
  is three call sites today and every future one would have to remember. A
  stale entry is not harmless, because this set exists to MEASURE how often
  disputes occur: a phantom inflates the figure the settlement decision turns
  on, and the report prints it as real. Same shape as `shard_in_backoff`, which
  self-evicts on read so the map stays bounded with no dedicated sweep and no
  obligation on code that has not been written yet.
- **The two outcomes are two events.** `verification_failure_event` and
  `verification_summary` are pure so their truth tables are pinned, because in
  both cases the branch existed in the code and was not carried through to the
  message the user reads:
  - The rescan emitted `shard_verification_failed` with a red error toast for a
    KEPT shard. `kept` was computed one line above and used only in the log.
    That key translates as "A model part failed its integrity check —
    re-downloading automatically" — false twice on this path — so the one case
    where the node deliberately stands by a possibly-last copy was reported as
    a fault with an automatic repair under way. The obvious action for the
    reader is to delete the shard, which is the destruction the policy exists
    to prevent.
  - The startup sweep's summary carried `verified`, `quarantined` and
    `unchecked` but not `disputed`, so a node keeping disagreeing bytes
    announced "Verified 20 shards". Same mistake as the one the `unchecked`
    counter was added to fix, one field later — a verifier that reports work it
    did not do reads as assurance.
- **Orange, not red.** The node is behaving correctly and deliberately;
  colouring it as a fault is what pushes an operator into the destructive
  action.

Settlement designs and the decision criteria: `docs/FUTURE_WORK.md` § "A
disputed shard is kept but the disagreement is never settled".

## The activity list reports a TRANSITION; the log may report every message

Three defects on 2026-09-14, all the same shape: a repetition lesson learned for
a **log line** and not carried to the **activity list**, which is the surface a
person actually reads.

The list is a 100-entry ring (`emit_activity`), and it is not only the Activity
panel — it is the replay a dashboard receives when it opens, and the "recent
activity" section of the pasteable diagnostics report. Anything emitted per
protocol message therefore does not merely add noise; it empties the ring of
everything the user did. Measured on the deployed .181 binary: **102 of 114
entries** were peers announcing parts.

The three, and what each already knew:

- **`shard_announced`** fired once per MODEL on every announcement.
  `note_build_tag`, one function below the holder code, had logged "ONCE per
  transition rather than once per announcement… a peer repeats itself
  indefinitely" since the build filter was written. Fixed by collapsing a
  multi-model announcement into one entry AND gating on a real change; the
  collapse is what bounds it, because a disconnect clears holder records so
  every reconnect is a genuine change for every shard.
- **`peer_connected`** fired on every Identify. `handle_identify_received`
  states at the top that "Identify re-pushes constantly" and gates its
  foreign-peer INFO on a set insert for that reason, then gates its "Peer
  connected" INFO on `connected_node_ids.insert` 350 lines later — and the
  activity event sat between the two, ungated. Three consecutive identical
  "Computer connected: …" entries for one computer was an ordinary sight. Fixed
  by moving the emit INSIDE the same transition gate as the log.
  Guard: `a_peer_connected_activity_entry_is_emitted_only_on_the_transition`,
  positional because that is exactly what the fix is.
- **The dispute count** is the inverse failure — see the section below.

**The rule for a new ActivityEvent:** ask what makes it fire. If the answer is
"a message a peer sends on a timer" or "a handler libp2p re-runs", gate it on a
state change, and prefer one entry per event over one per item inside the event.
The DIAG log beside it is the durable per-message record and is deliberately
untouched — `daemon::dispatch`'s shard-announce DIAG says so in its own comment.

⚠ **Measure this correctly.** A subscription that replays history on connect
makes the first ~100 messages look like live traffic; the first attempt here
recorded "1.9 events/second" for a node emitting **zero** live in 87 seconds,
and the wrong figure reached a commit message. Discard the connect burst before
calling anything a rate — and note that a rate and a burst want different fixes
(gotcha #613).

**In short** (the rule statement as it stood in `.claude/rules/arch-network.md`). `emit_activity`'s ring is 100 entries, and it is also the replay a dashboard
opens on and the "recent activity" of the pasteable report. An event emitted per
protocol message empties it of everything the user did — measured at 102 of 114
entries. Before adding an ActivityEvent, ask what makes it fire: if that is a
peer's timer or a handler libp2p re-runs, gate it on a state CHANGE and prefer
one entry per event over one per item inside it. Three sites had this wrong at
once, each with the identical lesson already written down for the log line
beside it. The per-message DIAG log is the durable record and stays.

## A count that reads zero while the thing happens is worse than no count

**`disputed_shards` exists so that "this node is serving bytes the swarm
disagrees with" is countable from outside the log**, because how often it
happens in the field is the open question that decides whether to build a
settlement protocol at all (`docs/FUTURE_WORK.md`). The diagnostics report
prints the section even at zero, deliberately: a pasted report saying `0` is a
measurement and one that says nothing is not.

That only holds if every path records. **Three compute `mismatch_policy` and can
land on `KeepBytes`** — `daemon::background::spawn_shard_verification` (startup
sweep), `auto_manage::scan::rescan_local_shards` (rescan), and
`auto_manage::manager::verify_pending_shards` (the drain of
`shards_pending_verification`, i.e. a held shard whose EXPECTED hash changed).
Until 2026-09-14 the third warned and returned.

**It is the path that matters most.** A node with origin provenance never gets
there: `register_manifest` refuses a contradicting claim outright, so no dispute
is created. The nodes that DO reach it are the ones whose manifests came from
peers — which is most of them, and precisely the population the count was added
to measure. Observed on an isolated node: two `A shard we are serving disagrees
with the hash the swarm reports — keeping our bytes` warnings for
llama-3.2-3b, beside its own report saying "shards kept despite disagreeing (0)
— none, every checked shard matches its expected hash".

It cost a real measurement. A reading of ZERO taken from the live node earlier
the same day had been recorded as evidence and had to be withdrawn, because it
could not be distinguished from "the path that would record it is not reached
here" (diagnosis rule 2 — absence of evidence from an incomplete source).

Three things a change here must keep:

- **The clear is unconditional on success**, on all three paths. Clearing only
  where a dispute is known is a clear that has to be predicted, and a predicted
  clear gets forgotten — the entry then outlives the disagreement.
- **Passing `KeepBytes` as a CONSTANT records nothing.** `model::acquisition`
  and `auto_manage::download` use it to ask whether a file is already on disk.
  That is a question, not an acceptance, and the guard keys on *computing* the
  policy precisely so those two stay out of scope.
- **Read through `disputed_shards_now`**, which self-evicts entries whose file
  has gone: a dispute also ends when the shard is deleted or pruned, and those
  call sites do not know about the set.

Guard: `every_path_that_keeps_disagreeing_bytes_records_the_dispute`, verified
to go red on the real pre-fix `manager.rs`.

## A holder record names a BUILD, not just a shard

**`ModelRegistry::shard_holders` filters out holders that positively claim a
different GGUF build**, against `expected_build_tag` — this node's own manifest
hash for that shard. It is the single read accessor for the holder map (~60
consumers), which is why the filter lives there and not at the call sites.

**Why.** A model id comes from a display name (`slugify_model_name`,
deliberately — it unified three disagreeing derivations, #310), so every
independent build of one model collapses into one identity. Three Q4_K_M builds
of Qwen2.5-Coder-7B were live on the swarm at once, within 800 bytes of each
other, sharing not one shard hash. Holder records keyed on `ShardId` alone
pooled them, so the scheduler routed to either: correctness was safe (every
shard is hash-verified before load) but a wrong pick is a guaranteed wasted
transfer of the whole shard. Measured live 2026-09-05 — one peer gossiped
contradicting hashes **556 times** while remaining a routing candidate **14,190
times** (gotcha #406).

Five things a change here must keep:

- **The tag is a per-shard CONTENT hash** (`build_tag_from_hash`), not the
  manifest hash the original design proposed. `merge_known_shard_hashes`
  recomputes a manifest hash locally whenever it recovers one the sender
  lacked, so two nodes on the same build routinely disagree on it — it would
  have produced false mismatches. Shard bytes have no such problem.
- **Our own manifest is the reference, and that is deliberately not a judgement
  about which build is correct.** Our hash is what a download would be verified
  against, so a holder that disagrees with it cannot hand us bytes we would
  accept, whoever is right. That is what makes the filter sound even when we
  have no origin knowledge.
- **Unknown never excludes**, on either side — `build_tags_conflict` is the one
  place that decides, so a caller cannot read "unknown" as "wrong". An older
  peer, a DHT provider record, a local registration and a partial holder's
  all-zero placeholder all land there. Same contract as
  `max_hostable_layers`, and what keeps a rollout routable.
- **A tagless re-announce must not erase a build already learned.** Peers
  re-announce on a timer, so one older-peer refresh would otherwise restore
  the pooling on the next tick.
- **The drop is reported.** `note_build_tag` logs once per *transition*, never
  per announcement — a peer repeats itself indefinitely (556 times here), and
  anything a peer repeats on a timer will be repeated at you for ever. A peer
  silently absent from every routing decision is otherwise undiagnosable from
  outside the process; `conflicting_build_holders` is the programmatic view.
- **Every count a person reads is the FILTERED one, and the drop is explained
  rather than merely applied.** Added 2026-09-14, after the live node showed
  the gap: `qwen2.5-coder-7b-instruct-q4-k-m` had two peers announcing parts of
  it, neither able to serve one, and the model card said three computers had a
  copy. The count came from `all_shard_entries`, which is documented as raw and
  whose own note allowed a consumer that renders "a count for a human" — but a
  count rendered to a human is a claim *to* that human, and this one picks the
  health sentence (`dashboard.say_safe` / `say_at_risk` / `say_only_you`) they
  read to decide whether the model keeps working. Two writers were wrong in the
  same way: `api::admin_models::listing` (`peers_hosting`) and
  `api::websocket::build_stats_message` (the per-shard `holders` in the 2 s
  `stats_update` tick) — and because both land in the SAME dashboard row, the
  unfiltered one did not merely overstate, it made the number depend on which
  writer touched the row last. Guard:
  `a_holder_count_shown_to_a_person_is_the_count_that_can_serve`, verified to
  go red on the real pre-fix `websocket.rs`.
- **"Has a part of it" and "could serve it alone" are two counts** (2026-09-21).
  `peers_hosting` counts a peer holding ANY part. That is right for "who
  contributes" — a split pipeline runs on exactly those peers — and wrong for
  the question a reader actually asks it. A plan logged four holders beside
  `total_standbys=0` and **both were true**: `find_standbys` needs a candidate
  covering the WHOLE segment, and two of those four held 7/9 and 4/9.
  `peers_complete` is BitTorrent's seeder/peer line, and is the count that can
  answer for the whole model.
  Measured on the live node the day it was added: **13 of 15 models** had more
  any-part holders than complete ones, and `thudm-glm-4-9b-0414-q4-k-m` read
  **five holders against one complete copy** — a reader given only the first
  number cannot tell a well-replicated model from a single point of failure.
  ⚠ **The dashboard's health badge was already correct** and was deliberately
  left alone: it is computed per SHARD, so `say_safe` is reached only when every
  part has ≥2 holders — a state in which the model genuinely is safe however few
  peers hold all of it. Swapping the count there would have understated it. The
  defect was the catalog, not the sentence. **Fourth firing of "a count is not a
  statement about X"** — #451 (coverage), #464 (capacity), #465 (per-segment
  availability). Guards:
  `a_model_listing_reports_who_has_a_part_and_who_has_all_of_it` plus a planted
  violation, and `complete_holders` is a pure helper so the vision sentinel and
  the empty-denominator case are unit-tested rather than argued.
  ⚠ **`u32::MAX` (mmproj) is stripped inside `complete_holders`, not by its two
  collection sites** — counted in the denominator it would leave every peer of
  every vision model incomplete for ever, and a rule two sites must remember is
  one a third will not.
- **A count that silently shrinks is worse than one that is too high**, so the
  peers dropped are reported next to the ones kept: `peers_other_build` rides
  beside `peers_hosting` in the model listing, and the card says "N other
  computers have a different version of this model, so their parts do not fit
  yours." `ModelPeerCounts` carries the pair precisely so a call site cannot
  supply one without the other. The one deliberate exception is
  `api::admin_hf::count_unique_shard_holders`, which answers "how many copies
  exist in the swarm" for a model the user has not downloaded yet — there a
  different build is still a real copy, and it is a no-op anyway because an
  absent local manifest means `BUILD_TAG_UNKNOWN`.

**`model::manifest::shard_announce` is the ONE constructor for
`ShardAnnounce`**, for the reason the "one invariant, N paths" rule gives: a
site that forgot the tag would send an announcement claiming nothing, which is
indistinguishable on the wire from an older peer — so a missed site is
invisible rather than merely wrong. Adding a field extends the helper, not the
eight call sites. `shard_announce_is_built_in_one_place` fails the build on a
bare literal.

**Switchable in the field**: `SWARMLLM_BUILD_FILTER=0` restores the old
behaviour. This is a swarm-wide protocol change and the filter can in principle
leave a model with no holders — correctly, since none of them could give us
bytes we would accept, but a node in that state used to reach the model via a
failed transfer and an origin refetch. The hatch is for diagnosing that without
a downgrade, the role `SWARMLLM_KV_RECONCILE=0` plays for #462.

**What it also closed, which the original write-up got wrong.** That entry said
"correctness is safe — every shard is hash-verified before load". True for a
model assembled on ONE node. In a distributed pipeline each node verifies its
own shards against its own manifest and **nothing compares the build between
segments**, so a chain could run layers 0-10 from one build and 10-20 from
another with both sides passing. Not garbage — both are quantisations of the
same model — but neither model's output, and silent. Closing it directly needs
a build discriminator carried per segment; that is the residual if the filter
is switched off.

**Still open**: the reverse index (`node_shards`) is unfiltered, which is
correct today because every consumer reads it about the LOCAL node. A consumer
that asked it about a peer would bypass this filter.

**From the rules file (moved 2026-10-02):**

**`ModelRegistry::shard_holders` filters out holders that positively claim a
different GGUF build**, against `expected_build_tag` — this node's own manifest
hash for that shard. It is the single read accessor for the holder map (~60
consumers), which is why the filter lives there and not at the call sites.

⚠ **"Holds a part" and "could serve it alone" are two counts, and the listing
reports both.** `peers_hosting` counts ANY part — right for "who contributes",
since a split pipeline runs on partial holders, and wrong for every question
about serving the model whole. `peers_complete` (BitTorrent's seeder/peer line)
is the one that can answer. A plan logged four holders beside
`total_standbys=0` and both were true; **13 of 15 models measured had more
any-part holders than complete ones.** Fourth firing of "a count is not a
statement about X" (#451 coverage, #464 capacity, #465 per-segment).
⚠ The dashboard health badge is per-SHARD and was already correct — do not
swap its count, which would understate a genuinely safe model.

**`all_shard_entries` is the RAW map, and a count rendered to a person is a
claim to that person.** Anything that shows or decides on a holder count asks
`shard_holders` per shard; narrowing to `contains(&local_node_id)` is the one
shape unaffected. Two writers of the same dashboard row had it wrong at once —
`peers_hosting` and the WebSocket `stats_update` tick — so the number changed
with whichever wrote last. **And the dropped peers are reported, not silently
subtracted**: `ModelPeerCounts` carries `servable` and `other_build` together
so neither can be supplied alone. Guard:
`a_holder_count_shown_to_a_person_is_the_count_that_can_serve`.

→ `docs/invariants/network.md`

## Completing libp2p Identify does not make a peer one of ours

**`network::manager::identify::peer_speaks_swarmllm`** is the single answer to
"is this libp2p node a SwarmLLM node?", and it is gated at the one point where
Identify turns a connection into a peer — BEFORE the Kademlia insert, so a
foreign node never enters the routing table either.

**The trap.** libp2p's `identify` protocol is `/ipfs/id/1.0.0` — universal to
every libp2p node on the internet, IPFS and every other rust-libp2p project
included. `handle_identify_received`
registered a peer on the strength of it alone: minting a `NodeId` from the
peer's Ed25519 key, establishing an encryption session, and counting it in
`connected_node_ids`, the number the dashboard shows. The field that carries
the peer's own statement of what it speaks — `info.protocols` — was never read
anywhere in the codebase.

Measured on the live swarm 2026-08-25/26 (gotcha #396): five foreign nodes on
Linode, port 4001, relaying through one another, appeared in every node's peer
list with no version, no shards and no latency but `healthy=true`. They
identify as `openhydra/0.1.0` on `rust-libp2p/0.45.0` — port 4001 is the
libp2p convention and identifies nobody, so reading IPFS into it was a guess;
the rejection log line reports `agent_version` precisely so the next one does
not have to be guessed at. Each one's
first real SwarmLLM request answered `OutboundFailure … The remote supports
none of the requested protocols` — the peer saying so in our own log, one
second after we adopted it. They arrive via **PEX**, which dials an address
without asking whose it is, so it spreads: a brand-new node with an empty data
directory acquired all five within 90 seconds.

**Both halves are load-bearing.** Declining to register is not sufficient:
`handle_connection_closed` re-dials any peer with no registry entry, on the
assumption it disconnected before Identify ran. So skipping registration alone
trades a wrong peer-list entry for an endless dial loop — worse, and silent.
`NetworkManager::foreign_peers` (bounded by `MAX_FOREIGN_PEERS`) records the
verdict, and BOTH that re-dial branch and the PEX dial loop consult it. A new
path that dials a peer learned from an untrusted source must do the same.

**Safe to gate** because `/swarmllm/id/1.0.0` has been the identify
`protocol_version` since the first P2P commit, so no released node fails it;
the protocol-list arm is the belt-and-braces for a future build that changes it.
Match the namespace as a PREFIX, never a substring — the peer controls those
strings.

**Declining to register is still not sufficient, and the second half needed a
third.** Not re-dialling was covered by `foreign_peers`, and every one of the
dial sites now routes through `NetworkManager::dial_checked`
(`every_dial_goes_through_the_foreign_peer_gate` in `tests/repo_consistency.rs`
keeps it that way) — **though "every" was wrong when this was written: the test
matched `self.swarm.dial(` and `discovery::bootstrap_peers` was a free function
taking `&mut Swarm`, so the one site that mattered was invisible to it for
another release (gotcha #405). The pattern is now both forms, comment lines
excluded, with the loopback probe named as the single exception.** And the node
*still* opened 5-6 connections per foreign peer
in seven minutes, each closed 43 ms later by this gate and then re-established.
`dial_checked` refused none of them across two runs, which is the measurement
that matters: **the dials come from inside libp2p, not from us.** Which behaviour
was never pinned down and no longer needs to be — `SwarmBehaviour::blocked_peers`
(`libp2p::allow_block_list`) refuses both directions at the swarm level, and
`handle_identify_received` blocks the peer before disconnecting it. Measured
5/6/6 connections → **1/1/1**, one unavoidable first contact each, healthy peers
unaffected.

**The general rule**: completing a handshake that everyone speaks proves nothing
about who you are talking to. Before treating a successful negotiation as
identity, ask which population could also complete it.

## One dial per PEER, never one per address

**`NetworkManager::dial_bootstrap_peers` + `discovery::plan_bootstrap_dials`**
are how bootstrap and cached addresses are dialled. Group by target peer, one
`DialOpts::peer_id(..).addresses(all)` each, through `dial_checked`, with
`PeerCondition::DisconnectedAndNotDialing`.

**Why per-address dialling is wrong, not merely wasteful.** A bare
`swarm.dial(addr)` carries no `PeerCondition`, so libp2p's per-peer dedup cannot
see it, and it does not reach `dial_checked`, so the foreign-peer gate does not
apply either. `discovery::bootstrap_peers` did exactly that, once per address,
on every discovery tick. A peer cached at two addresses (TCP + QUIC) therefore
got two simultaneous dials whenever it was momentarily disconnected; with
request_response's own dial that reaches `max_connections_per_peer = 3`.
The vendored rr layer then spreads sends across all three (ranked, but ranked
among connections that should not all exist), and one that has quietly died
swallows its share until the 8-failure rule closes the peer entirely. Measured
paired against an unpatched node, same swarm, same 43 minutes: **13 connection
establishments to one peer against 3** (gotcha #405).

**It costs nothing.** `libp2p_swarm::connection::pool::concurrent_dial` is a
`FuturesUnordered` — one dial attempt RACES every address it was given, bounded
by `dial_concurrency_factor`, and yields exactly one connection. Handing one
attempt every address is the same parallelism; it just stops keeping the losers.
Do not "restore" per-address dialling for latency.

Three rules a new dial site must follow:

- **Read the peer from the LAST `/p2p/` component** (`target_peer_from_address`).
  A relay circuit is `…/p2p/<relay>/p2p-circuit/p2p/<target>`; the first names
  the RELAY, so every gate then asks about the wrong node and the dial asks
  libp2p to reach the relay at an address belonging to someone else. This was
  wrong in two independent places.
- **Do not weaken the condition.** `PeerCondition::Disconnected` asks only
  whether a connection is ESTABLISHED, so two dials issued before either
  completes both pass. `DisconnectedAndNotDialing` is libp2p's own default and
  our code had explicitly opted out of it at three sites. **The mDNS site keeps
  the weaker condition deliberately** — a LAN rediscovery must be able to
  override a stale bootstrap attempt; that one has a documented reason and the
  WAN paths did not.
- **Never dial yourself.** `dial_checked` refuses it for every source. A third
  party hands your own address back — PEX relays whatever its registry holds,
  and a circuit terminating at you names you in its last hop — and the
  sender-side self-filter cannot help, because the sender is someone else.
  libp2p refuses these, so nothing broke; it was 223 wasted dials in 45 minutes
  that nothing surfaced.

**Dial attribution is logged at DEBUG** (`site` field in `dial_checked`), not
TRACE. Two investigations have turned on "was that dial ours?" and both had to
rebuild to answer it; the second only found the self-dialling because the
instrument was finally there to say so.

## Peer Cache: storable vs dialable

`network/peer_cache.rs` answers two different questions and they must not be
conflated:

- **`filter_storable(addrs, local_peer_id)`** — what is worth *keeping*. Drops
  only what is junk under any circumstances: not remotely reachable
  (`addr_is_remotely_reachable`), or routing through our own peer id in ANY
  `/p2p/` hop (the relay position of a `/p2p-circuit`, not just the target).
  **Keeps private addresses regardless of where this node currently is** — a
  laptop saving its cache on a hotspot must not permanently lose the LAN peers
  it had at home. Used by `save_peer_cache`.
- **`filter_dialable(addrs, local_peer_id, local_addrs)`** — what is worth
  dialling *from here*. Everything `filter_storable` does, plus a peer's
  RFC1918 / CGNAT / IPv6-ULA addresses are dropped when EITHER: (a) our own
  reachable addresses contain no private address (`local_is_public_only` — we
  can't route to anyone else's private network), OR (b) **the peer itself
  advertises a publicly-reachable address** (`peer_has_public` — then its
  private addresses are its own LAN/Docker bridge and we reach it publicly
  instead). Used by every dial path and by `GET /api/admin/diagnostics`.

  The `peer_has_public` clause is the **Docker fix** (2026-07-23): a Docker
  node advertises its container bridge `172.17.0.1` alongside its real public
  IP, and `172.17.0.1` is not globally unique — it is the Docker gateway of
  *whichever* host dials it, so a dial loops back to the dialer's own node
  rather than failing cleanly (confirmed live). A peer with a public address is
  reached there; its private noise is dropped even when we are on a LAN too.
  A peer with ONLY private addresses (no public) is still kept, so the home
  two-machine / pool case is untouched — those peers are additionally found via
  mDNS regardless.

**`local_addrs` empty means "not bound yet", NOT "public."** `listen_multiaddrs`
is empty until the swarm finishes binding; a node seconds into starting that
concluded it was a public server would discard every LAN peer it had, breaking
the home two-machine and pool cases the cache exists for. So the
`local_is_public_only` clause treats empty as unknown → keep. The
`peer_has_public` clause is independent of local context: it keys on the peer's
own addresses, so it correctly drops a public-capable peer's Docker/LAN noise
even at startup.

Nothing in `src/pool/` reads this cache — pools route through `pool_state` /
`allowed_node_set` — and mDNS discovers LAN peers independently, so a LAN pool
has a second route back regardless.

Retraction of a peer's *shard* claims is a different mechanism entirely; see
`ShardAnnounce.complete_for_models` below.

## ModelRegistry Holder Counts

**`merge_dht_providers` is the one writer of `shard_holders` that cannot remove
a holder** — it loops `record_shard_holder` over a DHT `GetProviders` result.
That matters because a provider record outlives the fact it asserts: libp2p-kad
keeps one for 24 h, republishes at 12 h, and other peers serve it, so a node that
deleted or lost a shard is still advertised as holding it for hours. An
add-only writer wins every disagreement with a writer that removes, and its
cadence decides how fast.

Measured live 2026-08-22 on a three-way split (gotcha #364): the holder
retracted shard 2 correctly and re-announced its reduced holding every 5 minutes
(`Peer retracted shards it no longer hosts … dropped=1`, six times), the
coordinator re-merged the stale DHT record every few seconds, and every request
was then scheduled onto a node without those weights — `503 Segment failover
exhausted`, indefinitely, while a healthy node holding exactly that shard was
never considered. **Retraction alone is futile when something re-adds the claim
faster than it is withdrawn** (same shape as #163).

So `ModelRegistry::retracted_claims` records what a holder has withdrawn, and
`merge_dht_providers` skips a (shard, node) pair found there. Two halves that
must stay together: `record_shard_holder` CLEARS the entry, so a node that
genuinely re-acquires the shard is believed the moment it announces that itself;
the DHT path must NEVER clear it, which is the entire point. Honoured for
`RETRACTION_HONOURED_SECS` (26 h — deliberately longer than the provider
record's own life, or the record simply wins again at the end of the window).

A new writer of `shard_holders` fed by anything other than the holder's own word
must answer the same question first: can it remove, and if not, what stops it
resurrecting something already withdrawn?

`ModelRegistry::shard_holders` caches at most `MAX_HOLDERS_PER_SHARD = 50`
holders per shard (LRU-evicted, local node never evicted). This is the
**routing oracle** — pipeline scheduler, region eviction, busy-holder
check etc. all read this map.

`ModelRegistry::global_holder_count` holds the **uncapped swarm-wide
count** from the most recent DHT `GetProviders` response, written by
`network/manager/dht.rs::handle_dht_providers_found` with the raw
`providers.len()` (PeerId count, not the resolved NodeId count — some
PeerIds may fail to resolve but they're still distinct providers in the
DHT's view). This is the **prune-score oracle** — `model/auto_manage/
prune.rs` uses `max(cached_holder_count, global_holder_count)` for the
`redundancy_ratio` numerator and the severe-saturation bonus check.

Don't:
- Read `global_holder_count` for routing decisions — DHT staleness is
  fine for an O(hundreds of seconds) prune cadence but unacceptable for
  scheduling.
- Read `shard_holders().len()` alone for `redundancy_ratio` — at 1000-
  node scale the cache pegs at 50 and the prune score saturates.
- Forget to clear `global_holder_count` when a model is removed —
  `remove_all_model_shards` retains over both maps; new code paths that
  evict a model must do the same or stale figures will inflate future
  ShardId-reuse scores.

## A report built to be handed to a stranger is a publishing surface (2026-09-01)

**`network::redact::redact_addresses`** hides every host in the diagnostics
report — an IP literal or a DNS name, in a multiaddr or in free prose — and is
called at the ONE point `api::admin::diagnostics` returns. `?full=1` is the
deliberate opt-in for an operator debugging their own machine;
`swarmllm diagnostics --full` and `examples/two_node_test.sh` are the two
callers that ask for it. `the_diagnostics_report_hides_addresses_unless_full_is_asked_for`
in `tests/repo_consistency.rs` fails the build if the default changes, and
checks the two surfaces that hand the report to a person still ask for the safe
form.

**Why it exists.** `GET /api/admin/diagnostics` has been the documented thing to
attach to a bug report for many releases; the dashboard has a one-click **Copy
diagnostics** button whose whole point is that a non-engineer does not have to
read what it copied; and its hint said *"No keys or invite codes are included"*,
which reads as *safe to post*. It also copied this machine's own addresses and
the first ten entries of the peer cache — on a live node, **other people's home
IP addresses**, verified against the real swarm (gotcha #426). A comment in
`reference-models.js` asserted the daemon had "already redacted" it, which is
the #419 shape: a stale assurance reads as verification and stops the next
person checking.

Four properties a change here must keep.

- **One pass over the finished text, not a rule per section.** Addresses reach
  the report from at least four places — the listen-address list, the peer
  cache, relay circuits, and the prose of whatever a failed dial produced — so a
  per-section rule is the "One invariant, N paths" trap below. A section added
  later inherits the redaction with no author action, pinned by a test that
  plants an address in free prose.
- **Keep the KIND, replace only the host.** Transport, port, peer id and the
  `/p2p-circuit` structure all survive, so "this node has no public address" and
  "that hop is relayed" stay legible. Deleting the lines would make `?full=1`
  the form people paste instead, which is worse than not redacting.
- **Salt the tag per report.** Two entries for one host share a tag, which is
  what keeps "ten cache entries, all one machine" readable. **IPv4 is 2^32**, so
  an unsalted digest is a reversible encoding of the thing being hidden,
  recovered by enumeration in seconds.
- **Exempt what identifies nobody.** Loopback and the unspecified address, and
  the project's own bootstrap anchor — the anchor ships in every binary, so
  hiding it protects no one and costs the most useful reading in the report. The
  exemption is derived from `default_bootstrap_peers()`, never restated, so
  moving the anchor moves the exemption.

**Peer ids and node ids are deliberately NOT redacted.** They are the swarm's
public identities, they appear throughout the rest of the report, and they are
not a coordinate anyone can dial. The address is.

**A query flag must have no invalid spelling.** `DiagnosticsQuery::full` is a
`String`, not a `bool`, because serde deserializes a `bool` from a query string
only for the literal words `true` and `false` — so `?full=1`, the form this
project's own README, `docs/DIAGNOSTICS.md` and `two_node_test.sh` all use, was
refused by the extractor with a bare-text 400 that never reached the JSON error
envelope (§ "API errors must be readable by the caller"). `query_flag_is_on` is
the shared predicate.

**The general rule.** Before writing that something is safe to share, enumerate
what is actually in it — and treat any surface built for a non-technical user to
hand to a stranger as a publishing surface, not a debug dump.

## `inference::pipeline::remote_generate::StreamReassembler`

(2026-08-09) — the
single place a remote reply's token stream is put back in order. Each token is
an independent `request_response` send, so the transport orders nothing between
them and the terminal token can overtake content still in flight. Emit through
the reassembler, never straight from the receive loop, and **never treat a
`finish_reason` as end-of-stream on its own** — that is precisely what
truncated replies from distant peers (gotcha #282).
The contract: content tokens carry `token_id` 0,1,2…; the done token carries
the total sent. An all-zero stream means the peer is too old to sequence, and
the reassembler degrades to arrival order so a mixed-version network keeps
working — do not "simplify" that away. Only the consecutive run is released, so
a lost token truncates rather than silently reordering the reply.
Any new multi-message exchange should be asked the same question — what happens
if these arrive backwards. R139's chunked activation forwards already answer it
(slot table indexed by `chunk_idx` plus a filled count); this path did not.

## A hole in a peer-served reply is FILLED, not waited out

(2026-09-02,
gotcha #438). `daemon::state::retained_replies::RetainedReplies` keeps each
fast-path reply this node streams — every content token as it is queued,
the terminal token when the decode ends — and `SwarmMessage::ResendTokens`
is answered from it, ONLY to the peer the reply was for. The requester's
`StreamReassembler` reports `has_hole` / `resend_range`, and the fast-path
loop asks after `hole_wait` (4×RTT, clamped 1-5 s), at most
`MAX_RESEND_ASKS` times, gated on the peer advertising
`features::RESEND_TOKENS`. Three things a change must keep: **a resend goes
only to the requester** (model output is the requester's and nobody else's);
**a duplicate of an emitted token is dropped by the reassembler**, or a
resend racing its original sits in `pending` for ever as a phantom hole;
and **the old deadlines still bound everything** — an exhausted ask budget
falls through to `STRAGGLER_TIMEOUT` / `INTER_TOKEN_TIMEOUT`, so a peer that
never answers cannot hold a request longer than before.
**An acknowledgement must come from the code that accepted the message.**
`requests.rs` used to send `SwarmResponse::Ack` for a `StreamingToken` its
own dispatcher had just dropped on backpressure — a drop reported as a
delivery. It now answers `SwarmResponse::Dropped` to a peer advertising the
bit (an older peer could not decode it and gets the ACK it always got), and
the serving side re-sends that token once from the retained reply, after
`STREAM_TOKEN_RESEND_DELAY_MS`; the re-send carries no `stream_token` key
in `PendingRrSend`, so it cannot loop, and anything further is the
requester's `ResendTokens` to ask for. `SWARMLLM_FAULT_DROP_STREAM_TOKEN=<n>`
drops content token `n` of every reply once — the only way to lose a token
on demand — and `SWARMLLM_RESEND_TOKENS=0` disables asking, so
`examples/dropped_token_test.sh` shows both the fix and the truncation it
replaces inside one binary.

## `NodeCapability.cpu`

(2026-08-18) — a processor described the way a graphics
card always has been. `GpuInfo` has existed since the beginning; the CPU had no
representation, so a peer without a card rendered as the bare word "CPU" and every
such machine looked identical. Additive and `#[serde(default)]`, per the
additive-protocol rule — verified in BOTH directions against the released
v0.3.101 binary: an older node ignores the new field with no deserialisation
failure, and a newer node reads `cpu: None` from an older one and falls back to
the old label. **A new capability field is not done until that pair has been run**;
the swarm is always mixed-version during a rollout.
Deliberately carries no more than the GPU already does — the `os` field's refusal
to send a build string is about identifying the INSTALL, whereas a processor model
identifies the hardware doing the work, which is what a peer needs to judge.

## `PeerInfo::ack_srtt_ms` is what routing prices a peer by

(2026-09-02).
Written by the network manager on every acknowledged tensor forward from
`AckRttEstimator::srtt_ms` (the RFC 6298 `srtt` the ACK deadline is built
from), capped at `ACK_SRTT_ROUTING_CAP_MS` (10 s) because the estimator
DOUBLES on a miss up to the deadline maximum — the right deadline for a
silent peer and the wrong latency for a route, which would take ~30 good
samples to decay. Read by `get_peer_metrics` ahead of `latency_ms`, the
health ping, which stays the fallback for a peer never forwarded to. The
ping is taken idle and cannot see the queueing a loaded event loop adds to
every forward (#386); routing prices the forward. **Local, never gossiped**
— it describes OUR path (the #341 rule). Pinned by
`a_measured_ack_latency_outranks_the_health_ping_when_choosing_a_holder`
with a control that the ping still decides without it.

## `mem_bandwidth::remeasure_keeping_the_best`

(2026-09-03 evening) — the
memory-bandwidth figure a processor-only node advertises may RISE over its
run and never fall. It was a `OnceLock` taken on the first capability
broadcast, so a node that booted busy carried a low figure for its whole
run — and since #428 that figure is what every peer's scheduler ranks it on.
Bandwidth is a hardware ceiling, so the best observation is the least
contaminated one (the argument `PASSES` already makes within one measurement).
The health monitor re-measures on a blocking thread at ten minutes and then
hourly, ONLY while no inference is in flight (a measurement under a decode
measures the decode), never on a GPU node. `best_of` treats an unmeasurable
pass as no information, not zero. A new consumer of the figure reads
`measured_gbps()` as before; a new measurement of any hardware ceiling should
be shaped the same way.

## A peer's advertised version may bring the update check FORWARD and may do nothing else

(2026-09-03 evening). `update::PeerVersionWatch` on
`state.events.peer_versions`, fed by the capability-gossip handler through
`EventBus::note_peer_version`; `state.events.update_nudge` (a `Notify`, not a
third broadcast channel — one listener, no payload) wakes `UpdateChecker::run`,
which then runs the SAME `check_for_update` the hourly poll runs, after a
random delay of up to 90 s and no closer than ten minutes to the last check.
The version is self-attested, so: two DISTINCT peers must agree; the version
must be newer AND plausibly adjacent (same major.minor, ≤ 25 patch releases
ahead); one version nudges once; a peer that reports something older
withdraws its vote; the map is capped. **Never let the gossiped value name,
select or fetch an artifact** — announcing `9.9.9` would otherwise be a
one-message way to make the whole swarm hit the update path at once.

## `update::SelfUpdateBlocker` — "this node cannot update itself" carries WHY

(2026-09-04, gotcha #450). `UpdateChecker::self_update_blocker` probes and
returns the reason; `can_self_update` is a thin wrapper over it. `key()` is
the stable string the dashboard translates (21 locales), `advice()` the
English one the daemon log and `swarmllm update` print — written together,
in one match, so a new case cannot reach one surface and not the others.
**Why**: it used to be a bare bool, and each of the three surfaces rendered
its own sentence about a package manager, correct for a `.deb` under
`ProtectSystem=strict` and useless to the Mac user whose binary sat in
`/Applications`. `UpdateInfo` now also carries `install_dir`, because the
folder is the thing the person has to act on and nothing named it.
**The CLI asks before downloading**: the probe is a file create-and-delete,
and `swarmllm update` used to fetch ~1 GB before discovering the answer.
`looks_like_packaged_install` keys on the unit file the packaging installs,
not on the binary's path — `/usr/local/bin` is equally a manual install, and
the advice that follows is wrong for the other case.

## `ModelRegistry::manifests_to_gossip`

(2026-08-11) — the single answer to
"which manifests should this node re-broadcast?": ones it published **and ones
it holds a shard of**. Both the one-shot startup announcement
(`daemon/background.rs`) and the 30s periodic broadcast
(`health/monitor.rs::broadcast_manifests`) go through it.
**Never filter on `publisher` alone.** Doing so broke model discovery
swarm-wide: every holder used to rewrite `publisher` to itself at startup to
earn broadcast rights, and `register_manifest` overwrites unconditionally, so
holders erased each other's claim until none of them broadcast. Since there is
**no on-demand manifest fetch**, a node that joined later could never learn a
model in full — `all_shards_available` stayed false and every request answered
"No model loaded" while the dashboard listed the model as available. Measured:
`phi-3.5-mini` registered 81 times under 50 distinct publishers (gotcha #296).
The correct predicate already existed in the startup path and was missing from
the timer, so discovery worked only for peers connected during someone's boot.
Holding a shard is the honest signal, which is why the gossip handler
deliberately does NOT require `sender == publisher`. `publisher` means who
published it — do not reintroduce a self-claim to grant broadcast rights.

## `model::manifest::keep_known_hashes_over_contradicting_ones`

**The rule.** When a gossiped manifest gives a different non-zero hash for a
shard we already have a non-zero hash for, and the two manifests describe the
same SHAPE, keep ours. Shards this node HOLDS are exempt.

**What it replaced.** `merge_known_shard_hashes` protected unknown → known and
its doc comment said the converse was "deliberately NOT protected: a real
incoming hash still wins over a real stored one". The reasoning was about
genuine re-publishes, and it was right about those and wrong about the signal
that identifies one: a re-publish is a new FILE, so its shard sizes and total
size move, and `ModelRegistry::describes_a_different_build` — which compares
SHAPE, never hashes — already sees it. A hash that changes while the shape does
not is not a re-publish.

**What it was measured at.** On the live node, 2026-09-18, v0.3.188-alpha, in a
71-minute window: **2,260 of 3,392 log lines — 67%** — were
`DIAG: register_manifest` for exactly three models, at 10-12 a minute each. The
INFO is gated on `changed`, computed below the merge and the origin override
precisely so a settled swarm stays quiet, so the stored `manifest_hash` was
genuinely flapping. Two peers re-gossiping different hashes for one file
overwrote each other indefinitely; the unit test read `[1, 3, 1, 3, 1, 3, 1, 3]`.

The three models were exactly the three the node held least of — 0, 0 and
1-of-16 shards. That correlation is the mechanism, not a coincidence: the origin
override (`origin_verified_hash`) settles any shard the node downloaded itself,
so only shards with no local provenance could flap.

**Why it is not only log noise.** `ModelRegistry::shard_holders` filters holders
by `expected_build_tag`, which is *this node's own manifest hash for that
shard*. An oscillating hash oscillates the set of peers the node believes can
serve those layers, so a request's candidate set depended on which half of the
flip it arrived in. It also cost a `persist` DB write and a `recheck` walk per
flip.

**What a change must keep.**

- **The shape gate.** Without it a genuine re-publish can never be adopted.
  `a_republished_model_still_replaces_the_hashes_we_had`.
- **The held-shard exemption.** For a shard on our own disk a contradicting
  hash is a *testable* claim, and adopting it is what makes `register_manifest`
  queue the re-check — the only way a node learns from the swarm that the bytes
  it is serving are wrong (gotcha #382; the startup sweep runs before any
  corrected hash can arrive). Stabilising those would trade a log flood for a
  corrupt shard nobody can report.
  `a_held_shard_is_rechecked_when_its_expected_hash_changes`.
- **Origin provenance still outranks everything.** The override runs after this
  and is unchanged. `an_origin_hash_outranks_a_peers_contradicting_claim`.
- **Blanks still learn.** `keeping_our_hash_does_not_stop_a_blank_being_filled`.

**What it does NOT claim.** Not that our hash is the right one. Only that
alternating between two unevidenced claims is worse than holding either, and
that an origin download — not whichever peer gossiped last — is what settles
it. The disagreement is now reported once, rate-limited on the same key the
origin-contradiction warning uses.

## `model::manifest::merge_known_shard_hashes`

(2026-08-24) — the rule that a
shard hash may go from unknown to known but never back. Called from
`ModelRegistry::register_manifest`, the single funnel every adoption path uses
(gossip ingress, DB reload, disk scan, acquisition), so no caller can skip it.
**Why**: a shard's BLAKE3 hash is a property of the MODEL, but a manifest is
built from what its author holds on disk — `build_shard_infos_from_layouts`
hashes a shard file only when it exists and writes all-zero otherwise. So every
partial holder publishes real hashes for its own shards and placeholders for
the rest, and the registry's blind `insert` let a placeholder destroy a hash we
already had. That matters because `network/manager/requests.rs` verifies a
completed P2P transfer ONLY when the manifest carries a non-zero hash: lose the
hash and the bytes are taken on trust, recorded as held, and re-served to other
peers unchecked. Measured on the live node — five shards fetched against a
manifest carrying placeholders for exactly those five, one corrupt
(gotcha #381).
Three things a change here must keep. The merge is **one-directional**: a real
incoming hash still replaces a real stored one (a genuine re-publish), and only
unknown is treated as no information — so this cannot be used to pin a stale
hash. `manifest_hash` is **recomputed ONCE, below EVERY correction**, because the
stored manifest is then a local composite rather than what the publisher sent
and `load_from_dir` re-derives that hash to validate a saved copy; recomputing
also keeps the changed-detection quiet, since each correction is deterministic
and an unchanged re-gossip lands on the same bytes.
**This used to say "when anything was recovered", and that narrowness was the
bug.** `register_manifest` corrects an incoming manifest TWICE — the merge
here, and the `origin_verified` override below it — and only the first
recomputed. So a manifest whose shard hash the origin had corrected said one
thing and authenticated another, and since `manifests_to_gossip` re-broadcasts
that exact object, every peer's `verify_hash_strict` refused it. The failure
was inverted: the better informed a node was, the more of its announcements
the swarm discarded. It was intermittent rather than permanent — the stale
hash is written only on the registration where a correction fires, and the
next registration needing none rewrites it consistently — which is why one
model is refused over and over in a log while others from the same sender
pass. The change-detection MUST read the final value too, or a peer
re-gossiping a contradicted manifest every 30 s compares a corrected hash
against an uncorrected one and reports a change for ever (gotcha #472).
**A hash OF a structure has exactly one compute site: after the last thing
that touches the structure.** The accept path is the consumer
that matters: `classify_p2p_shard_acceptance` (same module) turns "do we have a
hash?" into a three-way policy rather than a yes/no gate — verify against the
hash; or, with no hash but a reachable origin, discard the peer's copy and
fetch that shard from the ORIGIN; or accept-unchecked only when neither is
possible. Enforcing verification unconditionally was shipped once and
soak-caught, which is why the third case survives — but it is now *reported* as
unchecked rather than counted as verified. The second case is self-limiting,
not a retreat from P2P: the origin download hashes what it writes, so the
manifest gains the real hash and this merge then spreads it by gossip.
Three things it must keep. The peer is NOT penalised — it may have served
perfect bytes, and "cannot tell" is neither fine nor the peer's fault. The
fetch must actually **happen** before a copy is discarded, which is why
`AutoShardManager::complete_pending_shard_fetches` sits OUTSIDE the
`auto_manage.enabled` gate — the same distinction already drawn for
`try_idle_vram_unload`: that switch means "do not decide what to fetch on my
behalf", not "abandon a shard this node already asked for". **Never throw away
data you cannot replace.** And a recovered hash is PERSISTED — via
`ModelRegistry::set_persist_hook`, installed at startup with a `Weak`, in the
same shape as `set_ram_budget_provider` — because `load_from_db` is what
repopulates the registry at boot and the disk copy is the thing carrying
placeholders. Persisting is gated on the merge having changed something, or a
peer re-gossiping placeholders writes to the DB every 30s for every model.
The sibling half is `daemon/background.rs`: a shard with no hash is counted as
`unchecked`, never as `verified`. It had been counted as verified, so the sweep
reported "all shards OK verified=21" over five shards it had never hashed. **A
check that cannot run must be reported, not rounded up into the success line.**

## `types::slugify_model_name`

(2026-08-15) — the single derivation of a model
id from a human display name. It is what
`daemon::manifest::generate_and_register_local_manifest` registers, persists
and gossips, so resolving a name a user typed, building the model's directory
path, and announcing which models this node hosts must all arrive at the same
string or they are looking for a model nobody published.
There were **three** derivations and no two agreed. Two were near-identical
slugifiers differing on any character that is neither alphanumeric nor `-`/`.`
— one DELETED it, the other REPLACED it with `-` — so `Model (Q4_K_M)`
registered as `model-q4-k-m` and resolved as `model-q4km`; quant suffixes carry
underscores, so that is an ordinary GGUF name. The third was no derivation at
all: `health::monitor`'s capability announcement sent the RAW display name, so
a node that loaded a model with `-m` advertised holdings under an id no peer
could match to a manifest — **invisible as a holder of a model it was sitting
on**, and a phantom `shard_count: 0` entry in every peer's list (gotcha #310).
**The replace-and-collapse semantics are canonical because they made the ids
already on disk and in the DHT.** `the_shared_helper_still_produces_the_ids_
already_on_disk` reproduces the old manifest algorithm verbatim and asserts
agreement, so changing the semantics renames every user's models and goes red.
A new surface that turns a name into an id calls this; it must never grow a
second copy, and a raw display name is never a `ModelId`.

## `model::huggingface::is_trusted_publisher`

(R141) — canonical
curator-allowlist check for an HF `repo_id`. Splits on the first `/`
and case-insensitively matches the prefix against
`TRUSTED_HF_PUBLISHERS` (in `huggingface/watcher.rs`). Used by BOTH
the watcher's trust-promotion path (`promote_trust_for_known_sources` →
`min_downloads_for_repo` consumes the tiered 10k/100k threshold) AND
the wishlist scorer (`compute_wishlist` Candidate-row pass — flat +10
score bonus + `wishlist.why.trusted_publisher` why-tag). Any new
surface that needs to gate on "is this from a known-good curator"
MUST go through this helper rather than re-creating the allowlist —
the allowlist is a trust delegation and divergence creates a security
/ consistency gap. Adding a curator: append to
`TRUSTED_HF_PUBLISHERS` (one place); both consumers pick it up
automatically. Removing a curator (compromise, abandoned account,
loss of trust) requires the same one-place edit; do not soft-disable
via wrappers because the trust delta is a real security event worth
surfacing in the diff.

## `SharedState::resolve_connected_peer_id_bytes`

The resolver to use for
any message that `network::manager::relay::is_relay_eligible` refuses, i.e.
everything except `RemoteGenerateRequest` / `StreamingToken` /
`CancelInference`. For those direct-only messages "reachable" means
"connected", so the ungated `resolve_peer_id_bytes` hands back a target the
send path can only drop. **Gossipsub reachability is NOT request_response
reachability**: a peer relayed to us through the mesh is frequently
undialable, and `peer_id_map` is deliberately persistent across disconnects
(its only eviction is gated behind an 8,000-entry soft cap that never trips on
a small swarm). Replying to gossip via `peer_id_map` alone produced an
unbounded 30s loop of undeliverable sends — one departed peer, 45% of a
night's log volume (gotcha #220). `connected_node_ids` is the liveness oracle;
`peer_registry` is explicitly NOT, being preserved across disconnects for
reconnect. **When adding such a gate, re-check any `else`/fallback arm below
it** — the health-pong site had a `Broadcast` fallback that a naive `None`
would have turned into mesh-wide traffic every 30s, worse than the bug.

## `ModelRegistry::describes_a_different_build`

(2026-08-29) — is this
manifest the same FILE as ours, or another build wearing the same name? A
model id comes from a display name (`slugify_model_name`), so every
independent GGUF build of one model collapses into one identity: three Q4_K_M
builds of Qwen2.5-Coder-7B-Instruct were live on the swarm at once —
4,683,073,536 / 4,683,074,144 / 4,683,074,336 bytes — sharing not one shard
hash (gotcha #406).
**Compares SHAPE, never hashes**: a manifest carries a real `size_bytes` for
every shard whether or not its author holds it, but a hash only for the ones
it does, so every partial holder would read as a different build under a hash
comparison. **Gated on `has_origin_knowledge`** — our own origin download is
the only evidence that is not just another node's assertion, the same
adjudicator `origin_verified` uses — so a genuine re-publish (new bytes, new
shape, no origin copy of ours to weigh against it) still lands.
**Why it matters**: adoption is `manifests.insert`, last-writer-wins, so
before this the other build's `shard_count`, `total_size_bytes` and per-shard
sizes overwrote ours while `origin_verified` kept our hashes — a manifest
describing one file and authenticating another, and `size_bytes` is what
decides byte-range requests.
**Still open**: `record_shard_holder` keys on `ShardId` alone, so holders of
different builds are pooled and the scheduler will route to either. Bounded —
verification catches it, so it costs a wasted transfer, never a wrong answer.
Closing it needs a build discriminator on `ShardAnnounce`; see
`docs/FUTURE_WORK.md`.

**Both rejection messages share ONE rate limiter**
(`manifest::note_manifest_rejection` over `manifest::RejectionKey`), because
they are the same event at two granularities and a peer re-gossips on a timer
for as long as it is up. The manifest arm was rate-limited on 2026-08-26
(4709 WARN lines, 14% of a month's warnings); **the per-shard arm was missed,
and it is the worse of the two — it fires once per SHARD**, so one publisher
with an 8-shard model on a 30 s cadence produced 16-28 lines a minute
indefinitely, ~10% of the whole log, measured live 2026-08-29. The key
distinguishes the two kinds so neither can silence the other, and each shard
is its own key so eight genuine disagreements still get eight lines. A new
"we are ignoring what this peer keeps telling us" warning belongs behind this
limiter, not beside it: **anything a peer repeats on a timer will be repeated
at you for ever, so the first question about such a log line is what silences
it.**

## `model::manifest::is_backup_artifact_id`

canonical check for a
model id that is a copied-folder backup (`<model>.FULLBACKUP`,
`<model>.old`, `<model>~`, `… copy`) rather than a real model identity.
A model's identity must come from the model, not from whatever a
directory was called. `ModelRegistry::register_manifest` nets this at
the single point EVERY adoption path funnels through (gossip ingress,
DB reload on startup, local disk scan, acquisition) — so a backup name
can neither be stored, persisted, nor re-gossiped regardless of how it
arrived. Belt-and-suspenders explicit guards also sit at the network
boundary (`daemon/dispatch` ModelManifest handler emits a
`security`/`manifest_rejected` activity event + skips the auto-manage
wake; ShardAnnounce + RegionShardSummary skip backup ids so holder /
region counts stay clean without a manifest) and the local disk scan
(`daemon/startup.rs`, so an artifact is never persisted). New surfaces
that accept a model id from disk or the network MUST reject via this
helper rather than re-deriving the keyword list. The keyword list is
matched against the LAST dotted segment only, so a legit id carrying
dots from its source filename (`tinyllama-1.1b-chat-v1.0.q4-k-m`) is
never caught. The v0.3.10 disk-scan-only guard was insufficient because
a peer on an older build re-gossips the name straight back in.

## A holder claim means VERIFIED, not transferred

**Rule:** `.claude/rules/arch-network.md` § "A holder claim means VERIFIED, not
transferred".

### What happened (found 2026-09-17, by the probe v0.3.184 shipped)

A peer's withdrawn shard claim kept coming back: **3039 reinstatements over 9
days** on the live node, one peer having the same GLM-4 shard retracted **344
times** over five days at a median gap of 330 s, across only 21 restarts.
Persistent but bursty — present in 106 of ~120 hours, a 4x spike on 09-13, and
entirely quiet for hours at a time.

The chain:

1. `acquisition::maybe_broadcast_shard_progress` broadcasts the moment a
   download reaches `pct == 100` — unconditionally, since its threshold check
   carries an explicit `&& pct != 100` — and sends `DownloadState::Downloading`,
   because the BLAKE3 check has not run yet.
2. `auto_manage::download` broadcasts `Complete` only AFTER that check passes.
   So a download that fails verification emits the first message and never the
   second.
3. The receiver read `state == Complete || progress_pct >= 100` as completion
   and called `record_shard_holder`, which by design clears any retraction —
   a first-hand claim is supposed to beat a stale DHT record.
4. The peer's own next `ShardAnnounce` honestly omitted the shard, so
   `retain_node_shards_for_model` dropped it again. The peer retried on backoff,
   reached 100% again, and the claim came back. Forever.

**Why it hurts:** a reinstated claim is a routing candidate. The scheduler hands
a segment to a peer that does not hold those weights, and the request spends its
first-token deadline waiting on a node that was never going to answer — the
135 s silent-peer waits seen on the live swarm.

### What a change must keep

- **`Complete` is the only wire value that asserts holding.** `Verifying` and
  `Failed` exist in `DownloadState` but are never broadcast; a percentage is a
  statement about bytes.
- **The sender's cap is for the FLEET, not for us.** Every node released up to
  and including v0.3.184 reads `progress_pct >= 100` as completion, and they do
  not all update. `IN_FLIGHT_MAX_PCT = 99` means an older peer never sees the
  value that trips it. The cadence still tracks true progress — only the
  advertised figure is capped — so the broadcast threshold is unchanged.
- **A peer at 100% stays visible as a download** until it says `Complete`;
  `health::monitor::cleanup_stale_peer_shard_downloads` sweeps one whose
  percentage stops moving, so a failed verify cannot pin the entry.

### The diagnosis lesson, which cost more than the fix

The path had been **explicitly ruled out by reading** the day before — "shard-
download progress (no such messages at all)" — and that observation was true.
It searched for `Complete` messages, and the messages doing the damage say
`Downloading`. **When a handler fires on `A || B`, grepping for A tells you
nothing about B** (diagnosis rule 2: absence of evidence is only evidence of
absence from a complete source).

The measurement the round log had specified as decisive — group the collector's
`shard announce ingested` lines by node and look for one peer sending two
different `(seen, total)` pairs — could not have settled it either: the tuple is
`(shards in this announce, shards NEWLY recorded)`, not `(seen, total)`. Its
second element drops to 0 on every re-announce **by design**, so running it
produced 64 "divergent" pairs that were the instrument working correctly. Read
the emitter before trusting a field name recorded in prose.

**From the rules file (moved 2026-10-02):**

Nothing may record a peer — or this node — as holding a shard until its BLAKE3
check has passed. `daemon::dispatch::progress_claims_the_peer_holds_it` is the
single reading of an inbound `ShardDownloadProgress` and requires
`DownloadState::Complete`; a percentage never asserts holding, because
`acquisition::maybe_broadcast_shard_progress` reaches 100 while the bytes are
still unverified — it caps its advertised figure at `IN_FLIGHT_MAX_PCT` for
exactly that reason. `Verifying` and `Failed` are never broadcast, so `Complete`
is the only wire value that means held.

BitTorrent draws the same line: BEP 3's `have` announces a piece downloaded
**and verified**, never one that merely finished transferring. This repo has
paid for it twice before — gotcha #78 (compute the hash before registering as
holder) and gotcha #184 ("when a validity check exists in several places, find
the one that writes the claim others read"). A rejected artifact that is still
advertised is worse than one simply absent: it is a routing candidate that
cannot serve.

→ `docs/invariants/network.md`

## Destroying a shard we hold needs better evidence than a stranger's claim

**Rule:** `.claude/rules/arch-network.md` § "Destroying a shard we hold needs
better evidence than a stranger's claim".

### What happened (observed live, 2026-09-13)

A node restart, and eleven minutes later 507 MB of a perfectly good shard had
been deleted and re-downloaded byte-identical. The full chain, from one log:

| time (UTC) | event |
|---|---|
| 01:39:05 | `Registered manifest from local shard directory model=llama-3.2-3b-instruct-q4-k-m shards=4` — our own, from disk |
| 01:39:06 | our `expected_build` for shard 0 is still `c8e0f58e…`, the hash our file actually has |
| 01:39:07 | peer `e561df35` gossips a manifest for a different build. Shard **1**'s claim is refused — `Ignoring a shard hash that contradicts the one we took from the model's origin`. Shard **0**'s is **not**, and is adopted |
| 01:39:22 | background verification hashes shard 0, finds `c8e0f58e…` where the (now peer-supplied) manifest says `0c7f5223…`, and **quarantines the file** |
| 01:42-01:47 | `No reachable node holds layers 0-2 of llama-3.2-3b-instruct-q4-k-m`; P2P refetch dies at 150 MB, `Reconciled stalled acquisition → Failed` |
| 01:49:47 | `Fetching from the model's origin — no peer copy could be verified` |
| 01:50:06 | origin download records provenance; the guard **now** fires for shard 0 |
| — | the file that arrived hashes `c8e0f58e…` — **identical to the one deleted** |

Shard 1 survived and shard 0 did not because `origin_verified` only ever holds
shards this node itself fetched from the ORIGIN (`record_origin_downloaded_shard`,
and the repair path). Shard 0 had been acquired over P2P, so there was no record,
so `register_manifest` had nothing to refuse the claim with.

### How often — six for six, on one node, in 55 hours

The log that caught this held six shard quarantines across three models. Every
one shows the same signature: the quarantine happens FIRST, and the
"contradicts the one we took from the model's origin" warning for that same
shard appears **4-11 minutes later**, i.e. the origin knowledge arrived only via
the re-download the quarantine forced.

| model / shard | quarantined | first origin-contradiction warning |
|---|---|---|
| phi-3.5-mini / 2 | 18:38:17 | 18:42:57 |
| meta-llama-3.1-8b / 0 | 22:19:31 | 22:24:46 |
| meta-llama-3.1-8b / 1 | 15:09:01 | 15:19:04 |
| meta-llama-3.1-8b / 4 | 16:08:38 | 16:14:07 |
| llama-3.2-3b / 1 | 17:03:38 | 17:09:37 |
| llama-3.2-3b / 0 | 01:39:22 | 01:50:06 |

**Be precise about what that proves.** For all six, the hash that condemned the
file had **no origin backing at the time** — the rejection warning is emitted on
first sight of a given (model, shard, claimed hash), so an earlier one would
have appeared. So all six were destroyed on evidence that did not meet the bar,
and all six would be prevented now. For exactly ONE — 2026-09-13, shard 0 — the
deleted bytes are also *proven* to have been good, because the replacement
hashed identically. The other five may have been genuinely corrupt; their
pre-deletion bytes are gone and the question cannot be reopened.

Roughly one every nine hours, each costing a ~500 MB transfer and several
minutes of that model being unservable.

### Why this is gotcha #384 again, not a new class

`daemon/state/repair.rs` already states the loop exactly: *"the wrong hash
displaces the right one and the re-check quarantines our GOOD copy, refetches,
and judges the replacement against the same wrong reference, forever."*
#384's fix — persist origin hashes, load them before any manifest — is correct
and was not enough: **its coverage is "shards we fetched from origin", and every
other held shard is still exposed.** A defence that only protects the shards
that were never at risk is the shape to watch for.

### Why the fix raises the bar for DESTRUCTION rather than widening coverage

Recording "our disk copy looks self-consistent" as origin-verified would make
that field mean something it does not say, and would lock in a genuinely bad
copy. Instead the *destructive* action now requires the evidence:

- Keeping bytes that are genuinely bad is **bounded** — whoever downloads them
  hashes them against their own manifest, and `shard_holders` already filters
  holders by build tag, so a peer on a different build never routes to us.
- Deleting bytes that are genuinely good is **not** — on a small swarm it takes
  the last copy, and the gossip that caused it is still there to judge the
  replacement.

A disagreement is still worth settling — but **this change does not settle it**,
and an earlier version of this file and of the commit message said it did. That
claim was wrong and was caught by tracing the consumer rather than the producer:
`complete_pending_shard_fetches` begins by treating any shard whose file is on
disk as already repaired and clearing its mark, so marking a shard we are
keeping by definition fetches nothing. The quarantine path worked only because
it had removed the file first. The marking is therefore not done on the kept
path, and settling is tracked as open work in `docs/FUTURE_WORK.md` § "A
disputed shard is kept but the disagreement is never settled".

What the change does deliver is the half that was destroying data: the bytes
survive, and the node keeps serving them.

### What a change here must keep

- `verify_shard`'s `OnMismatch` stays a **required** parameter. It is what
  surfaced the seventh call site (`network/manager/requests.rs`) and the one in
  `model/acquisition.rs` that quarantined as a side effect of asking *"is this
  shard still missing?"* — a destructive answer to a read-only question.
- The three re-verification passes (`daemon/background.rs`,
  `model/auto_manage/scan.rs`, `model/auto_manage/manager.rs`) ask
  `mismatch_policy`; the two accept gates (`network/manager/requests.rs`,
  `model/acquisition.rs` post-download) pass `Quarantine` unconditionally.
- On `KeepBytes` the node also **keeps advertising** the shard. De-advertising on
  an unproven claim is the same mistake as deleting on it.
- The background pass counts `disputed` separately from `quarantined` and
  `unchecked`. Folding it into either would report work that did not happen —
  the same principle that made `unchecked` its own counter.

### Candidate explanation for a field report

Open item #49 ("auto-prune evicted shards the swarm had no other copy of")
reports a node that ended up missing **shard 0 of gemma-2-2b-it** and 4 of 5 of
phi-3.5-mini, with re-acquisition then looping `stalled shard download …
stall_secs=30`. That is the same shape as the run above — shard 0, a failed
refetch, a stall — reached without prune being involved at all. **This is a
hypothesis, not an attribution**: #49 is still blocked on the one log line
already requested from the reporter, and that line discriminates both stories.

**From the rules file (moved 2026-10-02):**

**`ModelRegistry::mismatch_policy` is the single answer to "may this node
quarantine its own copy of a shard whose bytes disagree with `expected`?"** —
yes only when `expected` is the hash we took from the model's ORIGIN.
`ShardStore::verify_shard` takes `OnMismatch` as a REQUIRED parameter, so the
four paths that re-check or merely QUERY bytes already on disk can no longer
inherit the accept-gate's behaviour by omission.

A gossiped hash is a claim about the claimant's build, not evidence about our
file. The asymmetry decides it: keeping bad bytes is bounded (the downloader
hashes what it gets, and `shard_holders` filters by build tag), while deleting
good bytes can take the swarm's last copy — and the same gossip is still there
to judge the replacement.

**The disagreement is NOT settled automatically** — the bytes are kept, the node
keeps serving and advertising them, and that is all. Marking such a shard for
repair does nothing (`complete_pending_shard_fetches` clears any mark whose file
is on disk), so it is deliberately not done. Settling it is open work:
`docs/FUTURE_WORK.md` § "A disputed shard is kept but the disagreement is never
settled".

→ `docs/invariants/network.md`

## One writer per shard file, and finishing one shard says nothing about the others

*Rule: `.claude/rules/arch-network.md` § "One writer per shard file, and
finishing one shard says nothing about the others". Field report, v0.3.178,
2026-09-13.*

### What was reported

A node auto-replicating an 11-shard, ~5.6 GB model saw its home connection
pegged at 10-15 Mbps with no inference running at all. Four shards landed; two
never did. The log showed, on a clean ~5-minute cadence matching
`auto_manage.interval_minutes`, for over two and a half hours:

```
HF layout drift detected (or sidecar missing) — discarding .tmp and restarting
shard=3 existing_bytes=33554432 sidecar_present=false
```

15+ full-shard attempts for the same 2-3 shards, each restarting from byte zero.
Reproduced independently on a second machine with a different config, against
the same model, failing on shards 3, 7 and 8. The reporter disabled
`auto_manage` on both nodes to stop it — which also cost them auto-pruning,
since it is the same switch.

### The mechanism

`max_concurrent_downloads` defaults to **3**, which is why 2-3 shards were
affected per machine and not one.

1. Three shards of the model download concurrently. They share ONE
   `acquisition_progress` entry, and its per-shard `Downloading` marks are what
   `is_shard_in_progress` reads.
2. The first shard finishes and calls `schedule_acquisition_cleanup`, which
   removed the whole model's entry five seconds later — unconditionally.
3. `is_shard_in_progress` now answers false for the two shards still being
   written. The next auto-manage tick re-selects them and spawns duplicates.
4. The duplicate finds a partial `.tmp` that does not sit on a coalesced-range
   boundary, so it deletes the `.tmp` AND its layout sidecar and starts a fresh
   one — while the original task keeps writing into the now-unlinked inode.
5. The original finishes its ranges, stats the `.tmp` *path* (now the
   duplicate's file), sees a size mismatch, and deletes that file and its
   sidecar as its own cleanup. Whichever ordering the two land in, one of them
   leaves a `.tmp` with no sidecar beside it — which is the `sidecar_present=false`
   in the report, logged for a sidecar the reporter could see on disk moments
   later, written by the next attempt.

Nothing here is specific to a shard index or a range count: shard 3 had
`ranges=2` and shard 7 `ranges=1`. It is specific to being still in flight when
a sibling finished.

### Why the guard was the wrong shape

`network/manager/requests.rs` already had the rule right, in a comment above its
own call: *"Remove the acquisition entry after a delay only when the entire
model is done — not after each individual shard."* Twelve other call sites did
not, which is this codebase's most-repeated defect — a shared invariant
implemented per path.

Deeper than that: exclusion between writers rested entirely on
`acquisition_progress`. That map is a PROGRESS structure — several subsystems
write it, the health monitor rewrites it, and a timer deletes from it. Using it
as a concurrency guard has now failed in the field twice: 2026-09-11 (a second
download request erased the marks of shards already in flight, fixed in
`begin_download`) and this one. The second fix is therefore not another patch to
the map but a claim that does not depend on it —
`ModelMgmt::shard_download_claims`, an RAII set written only by
`claim_shard_download` and cleared only by `Drop`, so a task that returns,
errors, panics or is aborted releases it with no cleanup call to forget.

`trigger_download` now *claims* where it used to *ask*, in one atomic step,
which also closes the window it had between asking and spawning.

### Two facts the P2P path adds

- **The `.tmp` is shared between transports.** HF writes packed tensor bytes and
  pins them with a `.tmp.layout` sidecar; P2P writes raw shard bytes at a chunk
  offset and resumes from `ShardStore::tmp_size`. An HF sidecar left beside
  P2P-written bytes describes a layout those bytes were not fetched against.
  Only the claim separates them.
- **The P2P claim is parked in `p2p_download_permits` beside the semaphore
  permit**, so the three places that release the permit (transfer complete,
  retry give-up, stall watchdog) release the claim too, by dropping the tuple.
  No new release site exists to be forgotten.

### What the prior art says

`huggingface_hub` hit the same class on its own `.incomplete` blobs and reached
a stronger conclusion: PR #4306 (merged 2026-06-05) **deleted cross-process
resume** and gave every attempt a unique `<etag>.<uuid8>.incomplete` +
`os.replace()`, because on Lustre/GPFS/NFS `flock` can silently succeed for
every caller, and *"sharing a partial file across processes is exactly what made
the corruption possible"*. Their locks now only save bandwidth; correctness does
not depend on them.

We keep resume deliberately, and the difference is why: their lock could be a
no-op on the user's filesystem, ours is an in-process set with RAII release in a
daemon that already owns its data directory exclusively (redb holds it). Keeping
resume matters here precisely because of the reporter's connection — a 512 MB
shard is 5-7 minutes at 10-15 Mbps, and losing resume means a dropped connection
costs the whole shard. If a second process ever shares a data directory, this
reasoning is void and the unique-name approach is the answer.

aria2's behaviour supplied the other half: it removes a control file whose data
file has gone (*"Removed the defunct control file… because the download file
doesn't exist"*). Startup `.tmp` cleanup removed the `.tmp` and left the
`.tmp.layout` behind, so the pair now move together.

### What a change here must keep

- A claim is released ONLY by dropping it. Never add a `release_claim` call —
  that is the cleanup path someone forgets, which is the whole bug.
- `model_has_live_shard_download` reads the claims and NOT the progress marks. A
  mark left behind by a path that gave up would otherwise pin the entry for
  ever, and the shard could never be fetched again — trading a tidy-up failure
  for an unfetchable shard. `is_shard_in_progress` reads both, because refusing
  to start a second download on stale evidence is the safe direction and
  deleting state is not.
- A terminal path must give its OWN shard a terminal progress mark. The HF
  failure arm did not, and was masked by the entry being deleted anyway.

**From the rules file (moved 2026-10-02):**

Every fetch of shard N writes the same `shard_NNN.bin.tmp`, and the HuggingFace
and P2P paths write it in different formats. Two writers corrupt it, and the
cleanup of whichever finishes first deletes the other's `.tmp` and layout
sidecar out from under it.

**`ModelMgmt::claim_shard_download` is the exclusion**, an RAII claim held for
the life of the download — moved into the spawned task on the HF path, parked
beside the semaphore permit in `p2p_download_permits` on the P2P one.
**`SharedState::remove_acquisition_if_idle` is the one place a progress entry is
removed for tidiness**, and it refuses while any shard of that model is still
being written; `a_finished_download_does_not_delete_the_progress_of_one_still_running`
in `tests/repo_consistency.rs` fails the build on a bare
`acquisition_progress.remove(`.

A per-shard completion path knows only about its own shard. Every one of them
was deleting the whole model's entry.

**Nothing but a download itself may delete that download's partial file.**
`cleanup_tmp_files_no_one_is_writing` skips any shard holding a claim; the
writer removes its own `.tmp` and layout sidecar together, which a directory
sweep cannot do. The startup sweep is the one unconditional one, and is correct
because at startup there are no writers.

**`ModelMgmt::live_cancel_flag` is the one source of a model's cancel flag**,
and every path that starts a download takes its flag from there — auto-manage
took none, so Cancel reported success and stopped nothing. A parked P2P
transfer HOLDS its flag (`P2pDownloadSlot.cancel`) rather than looking the model
up, because the map's entry is replaced when a download starts after a cancel.

→ `docs/invariants/network.md`

## Cancel means the same thing on both transports, and never takes a file from a live writer

*Rule: `.claude/rules/arch-network.md` § "One writer per shard file…". Found
2026-09-14 while fixing the restart loop above; not field-reported, but the same
collision reachable from a button.*

### What Cancel did

`POST /api/admin/hf/cancel/:model` (three entry points in the dashboard) did
three things, and two of them were wrong:

1. **It set a flag nobody was holding.** `download_cancel_flags` was written
   ONLY by `begin_download`, which only the API download paths call.
   Auto-manage registered nothing and passed `cancel_flag: None` into
   `download_shard`. So for an auto-managed download the endpoint's
   `if let Some(flag)` found nothing, set nothing, and returned
   `{"status": "cancelled"}` while the bytes kept arriving — on the connection
   the user was pressing the button to free.
2. **It deleted every `*.tmp` in the model directory.** The downloads it had
   just told to stop had not noticed (the flag is read once per chunk), so their
   partial files were removed from under them. That is exactly the collision
   that produced the restart loop: the writer continues into an unlinked inode,
   then stats a path holding somebody else's file, fails its size check, and
   deletes that one too. Pressing Cancel could start the loop.
3. It did nothing at all to a P2P transfer, which is a chain of
   request/response hops rather than a loop with a flag to read.

### The rules now

- **`ModelMgmt::live_cancel_flag` is the one source of a model's cancel flag**,
  and every path that starts a download takes its flag from there. One flag per
  model, because one press of Cancel is asking for everything being fetched for
  that model to stop. A flag that is already SET is never handed out — it
  belongs to a cancel in progress, and a fresh download given it would cancel
  itself the instant it started.
- **A live download cleans up its own `.tmp` and layout sidecar**, together.
  Nothing else may delete a partial file: `cleanup_tmp_files_no_one_is_writing`
  skips any shard holding a `ShardDownloadClaim`, so a sweep can only remove
  what a previous run left behind. The startup sweep
  (`cleanup_tmp_files_in_dir`) stays unconditional, and is correct to be — at
  startup there are no writers.
- **A parked P2P transfer holds its own cancel flag** (`P2pDownloadSlot.cancel`),
  it does not look the model up in the map. The map's entry is replaced when a
  download starts after a cancel, so a transfer consulting it could be shown a
  newer, unset flag belonging to a different download and sail straight through
  the cancel meant for it.
- **`P2pDownloadSlot` is where everything a parked transfer owns lives** —
  permit, writer claim, cancel flag, start time — so the four places that end a
  transfer release all of it by removing one entry. Adding something a transfer
  owns means adding a field, not another map with its own release sites.
- **A cancelled download is not a failed one.** The auto-manage error arm asks
  the FLAG, not the error text: a cancel recorded as a failure earns the shard
  an exponential backoff and the user a red toast for doing what they meant to.
  Asking the text would be gotcha #295 — that prose gets rewritten.
- **`abort_shard_transfer_if_cancelled` is deliberately NOT
  `retry_shard_or_fallback`.** The latter exists for a transfer that FAILED and
  its job is to find the bytes elsewhere, which for a cancel is the opposite of
  what was asked. It returns `false` (the download has ended) on the cancel
  path, never `true` (a retry is on its way).

## LAN membership is decided only from what we observed (2026-09-19)

**What it replaced.** `handle_identify_received` computed

```rust
let addr_is_lan = multiaddr_is_local(&info.observed_addr) || /* connection addr */;
```

`observed_addr` is what the PEER reports it sees our address as. It is
attacker-controlled, and it decided whether that peer sat inside our privacy
boundary: `pool::scope::allowed_node_set` (and `api/pool.rs`) admit every
`is_lan_peer` into private mode when `pool.private_mode_allow_lan` is on, which
defaults to `true` (`config/credit.rs`). A peer could therefore place itself
inside private mode by reporting that it observed us at `192.168.x.x`, and
outbound prompts would be routed to it.

**How it was found.** Not by the exploit — by a user noticing a peer listed as
LAN with nothing else filled in. Peer `9594e1ffaa2d8156`, nickname "win": public
IP `87.4.107.33`, **1220 ms** RTT, `is_lan_peer: true`, advertising a Docker
bridge address (`172.17.0.5`) and `127.0.0.1` alongside its real one.

**Two things made it hard to see.** The code already carried a comment saying it
deliberately did NOT infer LAN from `listen_addrs` — true, and reassuring, and
about a different input than the one that was wrong. And the log line read
`LAN peer detected from listen_addrs`, naming evidence the code had not used
since that comment was written. Both have been corrected; a rule that lives only
in a comment gets re-broken (gotcha #593), and here the comment actively
misdirected.

**Affected releases.** Introduced 2026-07-21 in `938e8de4`, so every release from
**v0.3.3-alpha to v0.3.191-alpha** inclusive. The flag is in-memory only and is
not persisted, so restarting on a fixed build clears any bad classification.

**What a change must keep.** The three legitimate inputs are ours: the
connection's remote address, mDNS, and a measured RTT. A relayed connection
carries no `ip4`/`ip6` hop at all, so `multiaddr_is_local` answers false for it,
which is correct. `an_asset…`-style scans are not enough here — the guard
inspects the statements that COMPUTE the decision (`let addr_is_lan =` /
`let is_lan =`), because the `PeerInfo` literal legitimately stores
`addresses: info.listen_addrs` in the same statement as `is_lan_peer: is_lan`.

⚠ **Still open**: the flag is sticky (`was_lan || addr_is_lan`, cleared only by
mDNS `Expired`), so a single spurious sub-5 ms RTT sample latches LAN for the
life of the process. No longer attacker-controlled, but worth revisiting.


---

**From the rules file (moved 2026-10-02):**

`is_lan_peer` is a privacy boundary, not a label: `pool::scope::allowed_node_set`
admits every LAN peer into private mode whenever `pool.private_mode_allow_lan` is
on, and that is the **default**. So its inputs are a trust decision.

**Never derive it from `identify::Info`.** Both `observed_addr` (the peer's claim
about what address it sees US on) and `listen_addrs` (its claim about itself) are
peer-controlled, and neither is evidence about where the peer is. The legitimate
inputs are the three we gather ourselves: the connection's own remote address,
mDNS (link-local multicast cannot be forged from off-link), and an RTT we
measured.

⚠ **The flag is sticky** — `was_lan || addr_is_lan`, cleared only by mDNS
`Expired` — so one wrong classification lasts for the life of the process.

Guard: `lan_membership_is_never_decided_from_what_a_peer_told_us`.

→ `docs/invariants/network.md`

## A network coordinate must be published rough, and fed a MINIMUM

**Rule**: `.claude/rules/arch-network.md` § "Network coordinates".
**Added** 2026-09-20, both halves found by deploying rather than by review.

### What it replaced

`NodeCandidate::latency_ms` is OUR round trip to a candidate. The routing cost
model had nothing else, so it priced each node in isolation and summed — unable
to distinguish three peers in one city from three on three continents. Measured
the same day: a 4-segment chain alternating Thailand↔Italy ran at **0.35 tok/s**
against **6.76** for the same kind of split inside 18 ms, and the difference is
entirely per-token traversals the model could not see.

Vivaldi (Dabek et al., SIGCOMM'04) gives each node a coordinate whose distance
to another node's coordinate predicts the round trip *between those two nodes*.
Implemented in `swarmllm_types::netcoord` as the paper's adaptive-timestep form
(Figure 3), 2-D plane plus height, `c_c = c_e = 0.25`.

### The two things that were wrong, and how they were found

**1. Publishing was gated on confidence, which deadlocks the whole system.**
`network_coord_for_publication` returned `None` until `is_usable()`. A node only
refines its coordinate against a peer that publishes one; every node starts
unsettled; so every node published nothing, nobody refined, nobody settled.
Symmetric and permanent. Observed exactly so: two nodes 4 ms apart, both on the
build, both reporting no prediction indefinitely.

A rough coordinate is not a hazard to publish — the error travels WITH it and
the receiver discounts every sample by `w = e_i / (e_i + e_j)`. That weighting
**is** the paper's mechanism for high-error nodes; withholding the coordinate
removes it rather than protecting anyone. `is_usable()` belongs to whoever
routes on a coordinate.

**2. The measurable round trip is not a network measurement.**
Fed raw samples, the coordinate learned the remote node's event-loop scheduling
delay. Measured against the LAN peer, 14 consecutive samples:

```
4, 118, 3, 3, 121, 132, 120, 123, 120, 120, 158, 132, 8, 125   (ms)
```

**Bimodal** — 3-8 ms or 118-158 ms, nothing between — against an ICMP round trip
to the same host, at the same time, of **min 0.563 / avg 0.920 / max 1.539 ms**,
with the remote at load 0.35. So ~120 ms of the application-level figure is the
peer's own scheduling, not distance.

`LatencyFilter` therefore answers with the **windowed minimum**, not each raw
sample. ⚠ **This deliberately differs from the reference implementation**:
HashiCorp's Serf/Consul keeps `LatencyFilterSamples` per node and takes their
MEDIAN, which is right for a light UDP gossip probe whose noise is modest and
symmetric. Ours is neither: 10 of those 14 samples are in the slow mode, so the
median IS the contamination (120 ms) and a median filter would encode it as
distance. A minimum is correct precisely when the corruption only ever ADDS,
which queueing and scheduling do — BBR's min-RTT argument, and the same
min-of-N discipline this repo already applies to benchmarking (gotcha #367).

The window is bounded in TIME (`LATENCY_WINDOW_MS`), not only in count, so a
path that genuinely degrades is not masked for ever by one good moment.

**3. A window sized in time is only as good as the rate that fills it
(2026-09-21).** The plan for this work was "soak, then judge `predicted_rtt_ms`
against the windowed minimum". Taking that reading on the release node found
something before the judging could start: **every peer reported
`rtt_samples: 3`**, against a `LATENCY_WINDOW_MAX_SAMPLES` of 64.

The arithmetic was never done. The only sample source that runs regardless of
load is the PEX ping, at `RR_PING_INTERVAL_SECS` = 120 s
(`network/manager/mod.rs`); the other, an acknowledged tensor forward, exists
only while this node is serving distributed work. So a merely-connected peer can
put **at most 3** samples in a 5-minute window — the cap was unreachable by a
factor of 21, and the two constants live in different files and were never read
against each other.

**That makes the minimum a minimum of three**, on an input where 10 of 14
observations are in the slow mode. Scored over every prefix of the sample above
at the real ping cadence, **8 of 14 answers land in the fast mode at three
samples per window; 13 of 14 with the floor** — the rest of the time Vivaldi is
taught the remote node's event-loop delay as the distance to it. That is a
plausible contributor to the near-peer overestimate the coordinate work already
recorded, though it is *not* established as its cause: the Azureus
closest-node limitation is a separate and sufficient explanation, and both can
hold at once.

`LATENCY_WINDOW_MIN_SAMPLES` is the fix — age a sample out on the ordinary
window only while more than this many remain. `LATENCY_SAMPLE_MAX_AGE_MS`
bounds it, so a peer that went silent for an hour and came back WORSE flushes
its stale window on the first new sample instead of answering with the minimum
it had back then.

⚠ **Raising the ping rate was the other option and was rejected**: an idle node's
upload was a live field complaint fixed the day before (entry 91), and more
probing is exactly what that fix removed. The floor costs a quiet link a slower
reaction to genuine degradation — ~16 min instead of 5 — which at one sample
per 120 s is the best available anyway.

### What a change must keep

- **Publish whenever we have a coordinate.** Pinned by
  `a_brand_new_node_still_publishes_its_coordinate`; verified by restoring the
  old behaviour and watching it fail.
- **Feed the filtered minimum, never a raw sample.**
  `SharedState::observe_network_coord` is the single writer and does the
  filtering itself, so no caller can bypass it.
- **Record the sample even when the peer publishes no coordinate yet** — the
  window describes the link, not our current ability to use it.
- ⚠ **If the measurement source is ever replaced by a true network-level round
  trip, revisit the minimum** — over clean samples a minimum chases the low tail
  and the median becomes the better estimator again.
- **Keep the window fed enough to be a window.** `LATENCY_WINDOW_MIN_SAMPLES`
  is paired with `RR_PING_INTERVAL_SECS`; changing either without the other
  puts the filter back where it was. Guard:
  `a_quiet_peers_window_usually_finds_the_fast_mode`, which scores every prefix
  of the real sample rather than its end — asserting on the last answer alone
  passes with the floor removed, because that sample happens to finish beside a
  fast observation (gotcha #502).
- Nothing routed on coordinates as of 2026-09-21. A consumer must check
  `is_usable()` and fall back to `region`.

**From the rules file (moved 2026-10-02):**

`NodeCapability.coord` is a Vivaldi coordinate so any reader can estimate the
round trip between two peers, **neither of which is the reader** — the one fact
the routing cost model never had (`NodeCandidate::latency_ms` is only ever OUR
round trip). Additive, `#[serde(default)]`, gated by `features::NETWORK_COORDS`.

Two halves, both learned by deploying rather than by review:

- **Publish it however rough.** Gating publication on `is_usable()` deadlocks
  every node simultaneously: a node refines only against a peer that publishes
  one, all start unsettled, so none ever publishes and none ever settles. The
  error rides WITH the coordinate and the receiver discounts by
  `w = e_i/(e_i+e_j)` — that weighting IS Vivaldi's answer to high-error nodes.
  `is_usable()` is the CONSUMER's gate. Guard:
  `a_brand_new_node_still_publishes_its_coordinate`.
- **Feed the windowed MINIMUM, never a raw sample.** The measurable round trip
  is application-level and carries the remote event loop's scheduling delay —
  measured bimodal at 3-8 ms or 118-158 ms against a peer whose ICMP round trip
  was ~1 ms. `LatencyFilter` answers with the minimum, and ⚠ **deliberately
  differs from Serf's median** because here the slow mode is the MAJORITY, so a
  median would encode the queueing as distance. `SharedState::observe_network_coord`
  is the single writer and filters internally so no caller can bypass it.
- **A window sized in TIME must be read against the rate that fills it.**
  `LATENCY_WINDOW_MS` (5 min) and `RR_PING_INTERVAL_SECS` (120 s) live in
  different files and were never read together: a merely-connected peer put 3
  samples in a window capped at 64, so the "minimum" was a minimum of three and
  answered in the slow mode a third of the time. `LATENCY_WINDOW_MIN_SAMPLES`
  is the floor that fixes it, `LATENCY_SAMPLE_MAX_AGE_MS` the bound that stops
  the floor resurrecting an hour-old estimate. **Before trusting any windowed
  statistic here, ask what feeds it and how often.**

→ `docs/invariants/network.md`

## Gossip volume: what an idle node was actually saying (2026-09-21)

`docs/FUTURE_WORK.md` #91 asked for the idle rate to be re-measured on a build
carrying the 2026-09-20 manifest fix before sizing anything else. This is that
measurement, and it moved the problem twice.

### The instrument came first

The byte counters alone could not answer *why* a topic was expensive, and
estimating message sizes against publish intervals produced an answer an order
of magnitude out — twice. GossipSub already keeps the counts; `GossipMeter` was
reading two of its six families. `network/bandwidth.rs` now parses all six:

| family | what it answers |
|---|---|
| `topic_msg_published` | what THIS node originated |
| `topic_msg_sent_counts` | that plus everything it forwarded |
| `topic_msg_recv_counts_unfiltered` | every copy the mesh delivered |
| `topic_msg_recv_counts` | what survived duplicate filtering |
| `topic_msg_{sent,recv}_bytes` | the bytes, as before |

`sent / published` is what relaying costs; `unfiltered / recv` is the duplicate
factor; `sent_bytes / sent` is the average message size. All three were guesses
before, and each guess was wrong.

### And a third time, deliberately this time (2026-09-21)

`swarm/models` is 82% of an idle node's upload, and it carries **six** variants.
`docs/FUTURE_WORK.md` #91's two remaining fixes — change-gating
`NodeCapabilityUpdate`, and moving manifests off the broadcast path — target
two *different* variants on that one topic, so the per-topic counters could not
rank them. **Estimating the split from sizes and intervals is precisely what
this entry records getting wrong three times**, so the instrument came first
again: `GossipKindMeter` counts received gossip by `SwarmMessage::kind_name`.

Measured on a probe node holding no models, 217 s window, 9 peers:

| variant | msg/s | KB/s | share | B/msg |
|---|---|---|---|---|
| `ModelManifest` | 2.93 | **92.51** | **86.4%** | **32,332** |
| `ModelDemandGossip` | 16.15 | 6.53 | 6.1% | 414 |
| `NodeCapabilityUpdate` | 1.08 | 3.65 | 3.4% | 3,477 |
| `RegionShardSummary` | 5.00 | 2.19 | 2.0% | 448 |
| `HfSourceGossip` | 3.01 | 1.21 | 1.1% | 412 |
| `ShardAnnounce` | 0.12 | 0.56 | 0.5% | 4,770 |

**The ranking is decisive and it is not close.** Manifests are 86.4% of inbound
gossip bytes; the capability broadcast is 3.4%, and change-gating it cannot
recover all of that because some ticks carry a real change. **The BEP 9 shape
is the fix worth building; `NodeCapabilityUpdate` gating is not**, and that is
now a measurement rather than a preference.

⚠ **A manifest averages 32 KB, not the 13 KB this entry and the work queue both
carried.** That figure came from the per-topic average over all six variants,
which the cheap ones drag down. **An average across a mixed population is not a
figure about any member of it.**

⚠ Note the highest message RATE — `ModelDemandGossip` at 16 msg/s — is 6% of
the bytes. Rate and volume rank differently, which is why both are reported.

### The same lesson again, six weeks' worth of confidence later (2026-09-21)

Six of ~28 families is still not all of them, and a tester found the next gap by
arithmetic: their node reported `swarm/models sent_bytes = 389.48 MB` against
`network_traffic.out_bytes = 178.9 MB`. **A part cannot be 2.2x its whole**, and
their marginal ratio held near 5x across three readings minutes apart.

**The counter is not wrong; its name is.** `msg_sent` is incremented at the
first statement of `send_message(peer_id, rpc)` — before the connected-peer
lookup, before the IDONTWANT check, and before
`peer.sender.send_message(rpc)`, which returns `Err` when that peer's handler
queue is full (libp2p-gossipsub 0.49.5). So `sent` means **attempted, once per
recipient**, and a forward dropped for a slow peer is counted and never sent.

Two consequences, both now written into the code:

1. **`sum(topic.sent_bytes) <= out_bytes` is NOT an invariant** and must not be
   asserted, which is what the reporter asked for. Their ask was right in spirit
   and wrong in mechanism: the gap is a measurement of dropped relaying.
2. **`GossipMeter`'s doc comment was the other half of the bug.** It claimed
   these counters were "the figure that matches what leaves the interface". The
   comment reasoned about deliveries while the counter measured attempts — the
   same trap § Timeouts records, where four of five constants had a comment
   about one quantity bounding another.

**Two drop paths, and only one has a metric.** Queue EXPIRY raises
`HandlerEvent::MessageDropped` and increments `publish_messages_dropped_per_topic`,
`forward_messages_dropped_per_topic` and `timedout_messages_dropped_per_topic`
(the last beside each of the first two, so summing all three double-counts).
Queue FULL bumps an internal `failed_messages` map and the peer score and
**touches no metric family at all** — it leaves the crate only as
`gossipsub::Event::SlowPeer`, drained per peer per heartbeat. A fix that read
only the metrics would have missed the half that actually moves on the
congested node that prompted the report.

**And the log could not have answered it.** The default filter is
`swarmllm=info`, scoped to our own crate, so libp2p's own `Send Queue full.
Could not send` WARN never appears: an 82 MB log on this node had **zero**
`libp2p_*` lines of any kind. That was nearly reported as "zero drops here" —
diagnosis rule 2, caught by asking whether the source could have shown it.

**The healthy reading, for comparison.** This node, same hour: gossip
2.674 GB against `out_bytes` 2.830 GB, ratio **0.94**, drops zero — the 5.5%
shortfall is Noise/yamux/QUIC framing. **A ratio above 1 is a node failing to
relay what the mesh hands it**, which is a capacity problem on that node and not
an accounting one.

⚠ **`sent_msgs` is not a publish rate**, and reading it as one is how
`swarm/regions` looked unfixed after v0.3.196 change-gated it. It includes
forwards times recipients. Measured on this node after the fix: **0.12
published/s against 236 sent/s** — the publish gate works, and essentially all
of the volume is relaying for a swarm still mostly on older builds.

### What it found

A probe node holding no models and serving nothing, on the live swarm:

```
topic          pub/s  sent/s  recv/s  unfil/s   dup   KB/s in   B/msg
swarm/regions   3.79  132.03   87.40   114.65  1.31      62.4      557
swarm/models    0.03    6.90    5.99     7.79  1.30      98.8   12,993
gossip total: 161.7 KB/s in, 157.4 KB/s out — ~2.6 Mbit/s to do nothing
```

Three conclusions, two of which contradicted the leading hypothesis:

1. **Duplicates were never the problem.** The duplicate factor is 1.31;
   GossipSub's deduplication was working. Tuning `mesh_n`, enabling IDONTWANT or
   disabling `flood_publish` would each have bought almost nothing.
2. **`swarm/regions` is a message-RATE problem**: 87 inbound per second at 557
   bytes. The probe itself published 113.7 messages per 30 s tick — ~93
   `ModelDemandGossip`, 20 `RegionShardSummary`, one wishlist — **about models
   it did not hold, carrying demand it had never measured.**
3. **`swarm/models` is a message-SIZE problem**: 13 KB per message, 6 per
   second, because a manifest carries every shard's full tensor table.

### The demand loop

`region_demand` is written by the inbound `ModelDemandGossip` handler AND
iterated wholesale by the publisher, which stamped `publisher: our_id` on every
entry. So each node re-originated the union of everyone's demand every 30 s, and
each round refreshed the timestamps the staleness check relies on — the entries
could not age out. A node that had served zero requests published 93 of them.

Fixed by splitting the two facts that shared one map: `local_region_demand`
(what we measured, keyed by model, maintained by `decay_request_counts`) and
`region_demand` (the merged view, still what scoring/pruning/wishlist read).
The publisher reads only the former. The key types differ, so the old code does
not compile at the publish site; `the_demand_we_gossip_is_the_demand_we_measured`
guards the rest of the file.

### The newcomer flood

`broadcast_manifests` treated "a peer we have not announced to is connected" as
a reason for a full BROADCAST round. Gossip cannot address one peer, so one join
cost every node in the swarm a full copy of every manifest. Measured directly:
restarting one probe took `swarm/models` inbound from 98.8 to **398.3 KB/s**,
and peers reconnect about 7 times an hour on an 8-peer node, so the trigger
fired roughly as often as the 5-minute periodic round — enough to about double
the full-round rate.

⚠ **An earlier version of this section said once every 80 s, and that was
wrong.** It counted `connection established` lines, which include the
post-restart dial burst (16 in one minute) and libp2p's several connections per
peer; a later sample was half OUR OWN probe node being restarted. Re-measured on
a window with no restart and no test node: **7 `io_error` closes in 57 minutes**
across 3 peers, 6 of them from 2, with dial failures bounded at 5 attempts and
exponential backoff. Note gotcha #46 — every peer fires a `clean_close` or
`idle_timeout` on KEEP_ALIVE expiry, so only `io_error` closes are a signal at
all — and `closed_findings.md` 08-29, which already settled peer churn as "that
peer's link, not our logic". The fix still stands: one join should not cost a
swarm-wide flood at any frequency, and the 98.8 → 398.3 KB/s spike per join is
measured. Its share of the steady state was overstated.

`NetworkCommand::SendDirectMessage` already carried any `SwarmMessage` over
request_response, and `requests.rs`'s fall-through dispatches it exactly as a
gossiped one — so the catch-up needed no new variant, no feature bit, and works
against peers that predate it. The periodic full round remains as the bound on a
catch-up that failed to land.

BitTorrent's answer is the same shape and older: BEP 3 sends the bitfield to the
peer that connected, over that connection, and broadcasts only per-piece `have`
deltas afterwards; BEP 9 fetches torrent metadata on demand in 16 KiB blocks,
verified against the infohash, rather than flooding it.

### Verified, not assumed

- Region summaries: published **3.79 → 0.08 msg/s**, the residual being the
  designed 5-minute anti-entropy round.
- The change-gate's null control: planting `timestamp_ms` into the digest turns
  `an_unchanged_region_summary_is_not_rebroadcast` red, which is the exact way
  this gate would silently stop gating.
- The demand guard's null control: the pre-fix read planted back into
  `health/monitor.rs` fails `the_demand_we_gossip_is_the_demand_we_measured`.
- The catch-up path: `DIAG: caught a new peer up on manifests directly` observed
  nine times with `delivered=1 of=1`.

### Still open

`NodeCapabilityUpdate` is broadcast every tick with no change gate.
`uptime_seconds`, `ram_available_mb` and `disk_available_mb` move every tick, so
it cannot be gated as it stands — the stable fields would have to be separated
from the volatile ones, or the volatile ones sent rarely. And a manifest at 13 KB
is still a broadcast payload; the BEP 9 shape (gossip `(model_id,
manifest_hash)`, fetch the tensor table on demand, verify against the hash) is
the remaining structural fix.

## The repetition, not the size: holders re-announcing each other (2026-09-23)

### What was measured

The release node on v0.3.200, 1 h up, 280 s window, zero inference, no downloads:

```
transport   42.2 KB/s in, 41.9 KB/s out  (~0.34 Mbit/s each way, ~110 GB/month)
gossip      39.9 KB/s received as delivered by the mesh, ~9 KB/s after dedup
by kind     ModelManifest      86.8%   0.22 msg/s   34,892 B/msg
            NodeCapabilityUpdate 7.9%   0.20 msg/s    3,617 B/msg
            everything else      5.3%
```

So after .196/.197 the manifest is still ~87% of what an idle node receives, but
the SHAPE of the cost is now visible: 0.22 manifests a second is **~66 per
5-minute round**, and in that window the registry logged no manifest that
genuinely changed. Every one was a holder re-announcing, on its own full round,
a manifest another holder had announced minutes earlier. Holders' round
counters are phased independently (by design, so they do not burst together),
so a model held by *k* nodes went out *k* times per round.

### What it replaced, and why not the tensor-table redesign

FUTURE_WORK #91's plan was to stop SENDING the tensor table (~92% of a
manifest), letting receivers derive it from the GGUF header. Reading the code
end to end showed it was more than "reorder one security check":

1. `verify_hash_strict` runs on gossip INGESTION too (`daemon/dispatch/mod.rs`,
   the `ModelManifest` arm), not only in `acquisition.rs` — the entry said
   registration did not verify. A node that hears of a brand-new model has no
   shards and no header, so it could not verify a table-less manifest there.
2. There is no peer-to-peer way to fetch `gguf_header.bin`; a node gets it from
   shard 0, the original GGUF or HuggingFace. (Loading needs it anyway, so
   deriving at LOAD time is always possible — the problem is only ingestion.)
3. GossipSub forwards the same bytes to every mesh peer, so "omit the table
   when every CONNECTED peer supports it" still reaches older nodes two hops
   away, which reject it (a rate-limited warning, no penalty) and stop learning
   those models. The documented way to change a gossip format is a new
   versioned TOPIC and a dual-subscription window — Ethereum's consensus layer
   puts the fork digest in every topic name for exactly this
   (`consensus-specs/specs/phase0/p2p-interface.md`: "Changing gossipsub/broadcasts
   requires a coordinated upgrade where all clients start publishing to the new
   topic together").

That is a protocol migration. Suppressing the repetition gets the same order of
saving with no wire change at all.

### The rule: RFC 6206 (Trickle) suppression

"A node that has recently heard exactly what it was about to say says nothing."
Trickle was designed for this — many nodes holding consistent data, each
tempted to re-advertise it on a timer — and its central parameter is the
redundancy constant *k*: transmit only if fewer than *k* consistent copies were
heard this interval. Here *k* = 1 and the interval is `MANIFEST_QUIET_WINDOW`
(30 min):

- `state.models.manifest_heard` records `(model → hash, when)` for a manifest
  the WHOLE SWARM was handed — a verified gossiped copy, or our own broadcast.
- `broadcast_manifests` skips any manifest whose exact hash is in there and
  recent, on a full round or as a "change" it merely learned from that gossip.
- A DIFFERENT hash is never suppressed: two holders disagreeing is information.

Two details decide whether it is safe, and both are enforced by construction:

- **Only GOSSIP proves the swarm heard it.** The point-to-point catch-up a
  newcomer is sent arrives as an ordinary `ModelManifest`. Counting it as heard
  would let reconnects (~7 an hour on an 8-peer node) keep every holder of a
  model quiet while the swarm never heard it. `AuthenticatedMessage.transport`
  is now a REQUIRED field, set at each of its four construction sites, and
  `note_manifest_heard` takes it as a parameter and ignores `Direct`.
- **Recorded only after `verify_hash_strict`**, so a forged copy carrying a real
  hash cannot silence the holders who would repair it.

It also closed a latent gap: a full round used to mark every connected peer as
"told", reasoning that the broadcast had just reached them. With suppression a
full round may broadcast nothing, so "told" now means only what the direct
catch-up actually delivered, retried each tick until it lands.

What the periodic round is still for, and why 30 minutes is enough: changes go
out the tick they happen; newcomers are caught up point to point; the registry
never expires a manifest that stops being repeated (`registry.rs` has no
manifest `last_seen` — its only staleness sweep is for shard-holder claims).
What remains is repairing a gossip message lost in flight, and that bound moves
from one round (5 min) to the window plus one round.

### Measured after (A/B, 2026-09-23)

Two isolated nodes (private `gossip_network_id`, no mDNS, no bootstrap), both
holding tinyllama, 40 minutes per arm, identical config — only the binary moved.
Metric: `gossip_recv_by_kind[ModelManifest].recv_msgs`, which on a two-node
network is exactly what the OTHER node gossiped.

```
                     received by A   received by B
v0.3.200                  11              11        (one more every ~5 min)
Trickle build              1               3
```

The Trickle build's sends, all of them: each node's one-time startup
announcement; A's first monitor round (07:42:50); and ONE re-announcement for
the pair at 08:14:11, A's first full round after the window lapsed — B's full
round came up the same second, had already heard A's, and stayed quiet. Every
other round on both nodes (144 of 146) logged `sent=0 suppressed=1`, so the
mechanism fired rather than the outcome changing for some other reason.
Scripts: `trickle_ab_v2.sh` (session e3b7669c scratchpad).

### Re-measured once the fleet carried it — and why it only half worked (2026-09-25)

Every connected peer ran ≥ v0.3.201. The release node, 57-85 min up, idle,
1660 s window:

```
                   v0.3.200 (09-23)          fleet on Trickle (09-25)
transport          42.2 in / 41.9 out KB/s   31.3 in / 23.5 out KB/s
gossip (mesh)      39.9 KB/s                 26.0 KB/s
ModelManifest      0.22 msg/s, 86.8%         0.083 msg/s, 74.9%, 40.6 KB each
```

2.6x fewer manifests, but 138 in 28 minutes where 19 known models predict ~19.
A probe node logging every received manifest (`DIAG: manifest received`,
debug, with each part's hash prefix) settled it: **every model had ONE
publisher, and 9 of 18 arrived in 2-5 VERSIONS** — the same parts carrying
different real hashes (no placeholders): holders that disagree about a part's
bytes, `docs/FUTURE_WORK.md` #61. Those nine were 100 of 111 and 63 of 64
gossiped copies in two windows (~90%); every model whose holders agree arrived
0-2 times in ten minutes.

**Mechanism.** `manifest_heard` held only the LAST hash per model. With two
versions in the swarm each holder's last-heard hash was always the other's, so
its own never counted as heard and it re-announced every full round, for ever —
the disagreement re-published every five minutes although receivers keep their
own hashes over a stranger's (`keep_known_hashes_over_contradicting_ones`) and
learn nothing from the repeat. RFC 6206 leaves "consistent" for the protocol to
define (its §6); here consistency is per VERSION. Keyed by `(model, hash)`,
each version goes out once per window swarm-wide and a disagreement is still
published. Test: `hearing_another_version_does_not_unhear_this_one` (red with
the old last-hash-only behaviour). **Expected after the fleet updates:** about
one manifest per version per window, ~0.02 msg/s where 0.083 was measured —
to be re-measured, not assumed. The tensor-table redesign stays deferred: it
would shrink each copy, while this removes the copies.

## A chained hop's refusal goes to the coordinator (2026-09-25)

**What happened.** The first live run of a composite failover
(`examples/split_rig.sh failover`) spliced C and D in for a failed last
segment, then — through a separate defect (`docs/invariants/scheduling.md`,
gotcha #706) — sent D a decode step for a conversation D had never been given.
D refused in 5 s ("no longer holds the conversation"). The coordinator heard
nothing and waited out the segment deadline, 290 s. D's log shows the refusal
sent; C's log shows it RECEIVED — by C, the previous hop of the chain, which had
no waiter and dropped it.

**Why.** `layer_forward.rs` has seven places that send a `LayerResult`. The
success path and the two chained-send failures computed `reply_target` — the
coordinator when the forward names one (`requester_node_id`, the `0x07`
trailer), the sender otherwise. The four early failures — no manifest, no local
shards, an invalid layer range, and the WORKER ERROR that carries every refusal
the worker makes — answered `sender_peer_bytes`. On an unchained forward those
are the same node, which is why nothing noticed; in a chain they are not. The
coordinator was ready for it: `PendingLayerResult::chain_members` accepts an
error from any hop of the run. Only the address was wrong.

**The fix, at the type.** `reply_target` returns a `ReplyTo` newtype that
nothing else constructs, and `send_error_result` takes one — so a reply
addressed to the raw sender no longer compiles. The handler resolves it through
one closure at every send, as the success path always did, so a connection made
while the worker computed is still used. Test:
`a_chained_hop_that_fails_tells_the_coordinator_not_its_predecessor` (red with
the closure returning the sender). The tensor-parallel partial is the one
deliberate exception: a TP forward is never chained, so its sender IS the
coordinator.

"One invariant, N paths" (`architecture.md`), in the network layer: the rule
"a result goes to whoever is waiting, not whoever handed you the work" was
written down (gotcha #354) for the success path and implemented on three of
seven.

**From the rules file (moved 2026-10-02):**

**`layer_forward::reply_target` is the single answer**, and it returns a
`ReplyTo` nothing else can build, so `send_error_result` cannot be handed the
raw sender. In a chain the sender is the previous hop, which is not waiting:
four of the handler's seven reply paths (no manifest, no shards, bad range, the
worker refusing) answered it, it dropped the refusal, and the coordinator sat
out its whole segment deadline — 290 s on the first live composite failover
(gotcha #707) — for a refusal the tail gave in milliseconds. The coordinator's
waiter already accepted an error from any `chain_members` hop; only the address
was wrong. The tensor-parallel partial still goes to the sender, correctly: a
TP forward is never chained.

→ `docs/invariants/network.md` § "A chained hop's refusal goes to the coordinator"

## The caller's sampling reaches a remote sampler (2026-09-25)

**What was wrong.** `LayerForward.sampling` was `#[serde(skip)]`: it carried the
request's temperature, top-p, top-k and penalties to a LOCAL last segment and
nothing to a remote one, whose worker then sampled at `SamplingParams::default()`
— 0.7 / 0.9 / 40, no penalties. Gotcha #399 had fixed the in-process half and
called the remote half "the honest state for a segment served on behalf of a
REMOTE coordinator"; that is the usual shape of a split (your computer holds the
first part, peers hold the rest), not an edge case.

**How it was found.** Not by reading: by scoring replies against llama.cpp
(`examples/score_against_reference.py`) on `examples/split_rig.sh repeat`. Four
greedy requests through one warm two-node split of llama-3.2-3b: the first takes
the n-gram path, which samples on the COORDINATOR, and was exact and repeatable;
the other three take the standard loop, which samples on the remote last segment,
and came back as three different replies with tokens the reference ranks 3rd or
4th by up to 2.05 logits. Reassociation (gotcha #370) is deterministic and moves
logits by hundredths — non-determinism plus off-rank choices is SAMPLING. A
teacher-forced rank is what separates "a near-tie flipped" from "this was
sampled", which #106 had been read as for a day.

**The fix.** A `0x0A` trailer (23 bytes) bound into the AAD through
`build_layer_forward_aad`, like every trailer before it. Gated on
`features::FORWARD_SAMPLING` at EVERY sender — the planned send, the failover
stand-in (gotcha #703: a second gate), and a chain head handing it to the next
hop; `peer_supports_pipeline_chain` requires the bit, so an older hop is never in
a chain where it would drop the parameters. The receiver clamps with
`api::clamp_peer_sampling` (gotcha #96's helper, extracted now that there are two
wire sources) — **after** rebuilding the AAD from the bytes as sent, because
clamping first rebuilds different bytes for any out-of-range value and fails the
seal, which reads as a key problem rather than a bad number.

**After.** The standard loop's three replies byte-identical, worst rank 2, largest
gap 0.41 logits. Remote `frequency_penalty` / `presence_penalty` work for the
first time: `0x08` (2026-09-21) shipped the history, but the values never arrived.
Tests: `the_callers_sampling_reaches_the_segment_that_samples_when_it_can_read_it`,
`a_hop_that_cannot_hand_down_the_callers_sampling_is_never_chained`,
`encrypted_envelope_carries_the_callers_sampling`,
`a_peers_sampling_is_clamped_after_the_seal_is_checked` — each red with its half
of the fix removed.

## A downloaded shard is hashed off the event loop (2026-09-25)

**What was wrong.** When a peer-served shard finished, `requests.rs` ran
`ShardStore::verify_shard` — a single-threaded BLAKE3 pass over the whole file —
inline, on the swarm event loop, which is the only consumer of every ping,
gossip message and tensor forward. `examples/split_rig.sh fetch` measured it on
v0.3.205: a 533 MB part of llama-3.2-3b logged `network event loop stalled …
arm=swarm_event:RequestResponse took_ms=206` at the moment it completed. Every
split this node was serving waited those 206 ms, once per part, and every
latency the node measured over that window carried them (`docs/FUTURE_WORK.md`
#108).

**Why it was not a one-line `spawn_blocking`.** It is the accept gate for
untrusted bytes, and both of its outcomes mutate state only the loop may touch:
quarantine, re-fetch, trust penalty and repair on a failure; holder
registration, the `ShardAnnounce`, the model reload and releasing the
download's claim on success.

**The shape.** libtorrent draws the same line: pieces are hashed on dedicated
threads and the verdict is posted back to the network thread. Here the hash runs
on the blocking pool and the verdict comes back through `shard_verdict_tx` — its
own channel, because the verdict carries a `SwarmError` (only a COMPLETE transfer
that hashes wrong implicates the sender) and `NetworkCommand` lives in the types
crate. `finish_p2p_shard` runs the rest on the loop. Three details the move
created: the download's claim stays parked until the verdict, so nothing else can
start writing the shard; a shard deleted while it was hashed is not registered;
and a hashing task that fails is our fault, never the peer's.

**After.** The same rig logs no stall. Guard:
`the_network_loop_never_hashes_a_shard_inline`, which flattens whitespace so a
rustfmt-wrapped call is seen, and was checked against a planted stray file.

**Its sibling on the DISPATCHER (#108b).** The manifest persist hook ran inside
`register_manifest`, which the dispatcher calls for every gossiped manifest, so
a redb commit — fsync included — ran on the only consumer of everything the
network delivers. The entry warned that a spawned write needs per-model
ordering: two quick updates must not land out of order. The answer is not to
order the writes but to make order irrelevant: the hook marks the model dirty,
and one background writer writes the registry's CURRENT record for it. One
writer means no concurrent pair; reading at write time means whichever update
came last is what the next write reads. With no runtime (a synchronous caller)
it still writes inline — there is no loop to stall. The same shape as the
nickname persist on 2026-09-24.

## A result names the step it answers (2026-09-25, #113)

**The failure.** A serving node computed a 28.8 MB prompt-pass result and its send
failed (`Tensor result fallback OutboundFailure — upstream will timeout`, field
report 3 of 2026-09-25, then quinn#2809's connection killer). The node KNEW; the
coordinator did not, and waited out its segment deadline before failing over.
The peer was still connected over another link, so one resend would have
delivered it.

**Why a plain resend was unsafe.** Every forward of a request to one segment
carries the same request id and the waiter was keyed by id and pinned node only.
If the first copy HAD arrived and only its acknowledgement was lost, the
coordinator would already be waiting on the next step — and the resend would
resolve that wait with the previous step's activations. No error; a wrong reply.
This is the at-least-once delivery problem, and the fix is the one idempotent
producers use (Kafka's producer id + sequence number): the receiver matches on
(request, step) and drops a copy of a step it has passed.

**What the step is.** `(index_pos, layer_range)`, not `sequence_num`:
`sequence_num` is 0 on the prompt pass and 1 on every forward after it
(`work_kind_for`), while within one attempt a segment's `index_pos` only grows —
prompt pass at 0, then each decode or verify step at its position. The position
ALONE was the first cut, and a reviewer found the hole the same night: the
planner may give one node two segments of one pipeline (A 0..10, B, A again
15..20), and every failover replay is sent at 0, so a resent copy of one
segment's answer could complete the other segment's wait at the same position.
The range closes it; a chained run's waiter admits every hop's range, since the
tail answers with its own and any hop may refuse with its own. A KV-truncate forward is fire-and-forget (no waiter).
A router retry restarts at 0 with the same id, but a copy from the abandoned
attempt at a matching position carries the same computation. The failover replay
is sent at 0 and its waiter expects 0.

**Wire.** `0x07 | index_pos, layer_start, layer_end` (u32 LE each) after `0x06`, set on every result a forward
produces, failures included. Harmless to an older coordinator without a gate: a
result is not sealed, and its decoder returns after the last trailer it knows and
never reads further — pinned by
`the_step_a_result_answers_rides_after_everything_an_older_decoder_reads` (frame
with = frame without + 13 bytes). The RESEND is gated
(`features::RESULT_STEP`), because only a coordinator that checks the step may be
sent a copy.

**Verified on the rig** (`split_rig.sh repeat`, llama-3.2-3b, A holds shard 0 and
coordinates, B the rest; B wrapped with `SWARMLLM_FAULT_RESULT`, which parks a
result and resends it on the 10 s sweep through the same `resend_lost_result` an
`OutboundFailure` calls):

| arm | B resends | A refused as an earlier step | router retries | reply vs control |
|---|---|---|---|---|
| control | 0 | 0 | 0 | — |
| `duplicate` (every result also resent ~10 s late) | 71 | 70 | 0 | byte-identical |
| `lose` (first eligible result withheld) | 1 (2 s later) | 0 | 0 | byte-identical |

Mixed versions through `split_rig.sh split`: a v0.3.206 coordinator with this
server, and this coordinator with a v0.3.206 server, both 200 across two segments.

**Limit.** The serving node arms a resend only once the coordinator's capability
gossip has arrived, so the first results of a pair that just met are unprotected
— the same limit as every feature gate in this protocol.

**From the rules file (moved 2026-10-02):**

A request's forwards to one segment share its id, so a result matched by id alone
cannot tell step N from N+1 — a late or resent copy would answer the NEXT step,
silently. `LayerResult::answers_step` (the `0x07` trailer, LAST but for a streamed answer's `0x08`) names the
forward — position AND layer range, because one node can serve two segments at
one position; `PendingLayerResult::expects_step` (`ExpectedStep`) refuses any
other. **Every registration sets it from the forward it actually sends** (every
hop's range for a chain; a failover replay waits on 0). `sequence_num` is a
prompt-pass flag, not a counter. Only then may a serving node
resend a result on `OutboundFailure` (`resend_lost_result`, gated on the
coordinator's `features::RESULT_STEP`, once). Test with `SWARMLLM_FAULT_RESULT=lose|duplicate`.

→ `docs/invariants/network.md` § "A result names the step it answers"

## Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind (2026-09-26)

**What it replaced.** The dispatcher bounds work for peers with a node-wide
semaphore (8 / 24 / 64 by contribution level) and a per-peer count (half of it,
floored at 4). Three defects in how:

1. **A refused forward was dropped, silently.** `continue` after a `warn!` — but
   the network manager had already answered the `LayerForward` with
   `SwarmResponse::Ack` on arrival (§ "A tensor forward is acknowledged on
   receipt"), so the coordinator knew the peer had it and waited for a RESULT: the
   whole segment deadline, for a refusal made in microseconds. A refused
   `RemoteGenerateRequest` cost the first-token wait the same way. The same shape
   as #707 (a chained hop's refusal sent to the wrong node): the right decision,
   never delivered.
2. **Only `LayerForward` was counted per peer.** Whole-model generations and image
   encodes took the node-wide semaphore alone — and a generation holds its permit
   for the entire reply — so one peer could hold every permit for minutes.
3. **The count's release removed its entry unconditionally** once it read ≤ 1, so
   a slot taken between the decrement and the remove lost its count.

**Now.** `PeerWorkSlot::try_take` (add, check, undo on overshoot) is taken by all
three kinds of peer work and given back by `Drop`; its release removes the entry
only while it still reads zero (`remove_if`, under the map's write lock). A
refused forward is answered by `layer_forward::refuse_forward` — addressed through
`reply_target` like every other reply, stamped with the step — and a refused
generation by `remote_generate::refuse_request`, which repeats the terminal frame
like every refusal on that path. The refusal carries only the address (never the
forward's activations), since one is spawned per refusal.

**The wording is the contract**: `peer_work_refusal()` renders through
`SwarmError::ServiceUnavailable`'s Display, which the coordinator reads as "this
peer cannot serve" (`router::message_means_peer_cannot_serve`) — bar it from this
request's retry and re-plan — and does NOT read as a missing shard, so the peer's
holder claims stay. Pinned by
`a_refused_forward_is_answered_to_the_coordinator_naming_its_step` (red with the
send removed).

**Research.** Google's SRE book, "Handling Overload": an overloaded backend should
reject cheaply and explicitly so the request is retried on another backend, and
keep per-customer limits. Its "overloaded; don't retry" variant exists to stop
retry storms; not needed here, where the coordinator re-plans once with the
refusing node barred.

**What is still open.** An image-encode refusal cannot be said —
`VisionEncodeResponse` has no error field — and admission is per MESSAGE, so a
split request's later token steps compete with new requests on every step
(`docs/FUTURE_WORK.md` #123). Incidence before the change: 0 refusals of any kind
in 9 days of the live node's log.

**From the rules file (moved 2026-10-02):**

A forward was acknowledged on arrival, so dropping it at the dispatcher's caps
cost the coordinator its whole segment deadline. **Every kind of peer work takes
a `PeerWorkSlot`** (forwards, whole-model generations, image encodes — generations
were uncounted and could hold every permit), and a refusal is **answered**:
`layer_forward::refuse_forward` / `remote_generate::refuse_request`, worded by
`peer_work_refusal()` so the coordinator bars the peer and re-plans without
retracting its shards. A spawned refusal carries the ADDRESS, never the payload.
An image-encode refusal still cannot be said (no error field; #123).
**A request that has not started keeps its hands off the last quarter of the
slots** (`admits_a_new_request`) — those are for running requests' next steps.
Real per-request admission needs an end-of-request signal first (FUTURE_WORK #123).

→ `docs/invariants/network.md` § "Work a serving node will not run is refused OUT LOUD"

## A newcomer is told our capability when it is identified, not at the next broadcast (2026-09-26, #120)

A coordinator that has heard no `NodeCapabilityUpdate` from a peer computes
`max_hostable_layers = None` for it, which every routing rung treats as
unbounded (`docs/invariants/scheduling.md` § "The relaxation is scoped"). A peer
is in that state from the moment it connects until its capability first arrives
— by broadcast on its own 30 s health tick, or relayed out of the gossip mesh's
message cache once the mesh forms. Live, 2026-09-26 04:06: a plan made 0.7 s
after a connection handed a whole 14B to such a peer (`max_hostable_layers=None`,
`est_tokens_per_sec=0.0`), which refused it for memory.

**Now** the identify handler, at the "Peer connected" transition, sends
`state.local_capability` — the exact capability this node last broadcast — to the
newcomer over request_response. An existing variant on the existing direct path
(the manifest catch-up's route), so no feature bit; the receiver's handler is
update-only, and our identify reaches it before this message does. If it does
not, nothing is lost: the next broadcast covers it, as before.

**Measured** (two throwaway dev nodes beside the swarm, no models, auto-manage and
updates off, default gossip network so the live v0.3.208 node's broadcast is the
control): between the two nodes carrying the change, the capability arrived
**0.2 ms and 10 ms** after "Peer connected"; from the seven peers on v0.3.208 it
arrived **4.9-5.4 s** after — all within one gossip heartbeat, i.e. from the mesh
cache, not the 30 s broadcast. Cost: one message per new connection (~7 an hour
on an 8-peer node).

### Running requests keep a reserve (2026-09-27, #123)

`dispatch::admits_a_new_request(available, total)`: a forward that starts a
request (`sequence_num == 0`), a whole-model generation or an image encode takes a
slot only while more than `max(1, total / 4)` are free. Under load the node-wide
cap used to refuse a running reply's next step like any newcomer. A sender can
claim to be mid-request, and that only reaches the reserve — both caps still bound
it. Per-request admission proper needs an end-of-request signal; why, and in what
order, is in `docs/FUTURE_WORK.md` #123. Test:
`a_new_request_leaves_a_reserve_for_running_ones`.

## A split token crosses on the pipeline stream, one per (request, peer) (2026-09-27)

**`inference.persistent_pipeline_stream` is measured 2.4x faster on a real link
and stays OFF by default** until the stall below is root-caused. The standard distributed
loop sends each forward on a `/swarmllm/pipeline/1.0.0` stream opened once per
request and peer (`network::pipeline_stream::PipelineStreamClient`), and the peer
answers on it; request-response is the per-forward fallback.

**Measured, WAN** — the spread benchmark, qwen2.5-coder-7b split this node
(RTX 3070, Thailand) L0-14 → a peer (RTX 4050, Belgium) L14-28, one peer 412 ms
away (app-level minimum), A-B-A inside one binary with only this flag changed:
request-response 1.19 tok/s (838 ms/token), stream 2.85 (351), request-response
again 1.20 (832). Verified per result, not by the clock: in the stream arm 0 of
64 results came through the rr dispatcher (`dispatcher received LayerResult`),
64 of 64 in the others. Per-token remote segment over rr: min 669 / p50 766 /
p90 807 ms — each rr message opens a substream. It had been left off after a
LOOPBACK measurement (no win where a round trip is free), which is the wrong
place to measure a per-message cost that scales with distance.

**The key is (request, peer), and it was request alone.** `send_forward` reused
any open stream for the request whatever `peer_id` it was given, so a plan the
coordinator drives across two remote peers would have sent the second peer's
forward — sealed for the second peer — down the first peer's stream. Found by
reading the client before flipping the default (the module doc had always said
"per (peer, request_id)"). `close(request_id)` now drops every stream the
request opened. Test: `a_request_keeps_one_stream_per_peer_and_closes_them_all`.

**Why it is not the default (FUTURE_WORK #133).** With the stream on, the 4-node
composite rig (`split_rig.sh failover`) had a HEALTHY peer B log `pipeline stream
handler started` and then nothing for ten minutes: it never finished reading
A's 1.8 MB prompt-pass frame, never dispatched it, and neither side logged an
error or closed a connection (three direct A↔B connections, no relay). The
request-response path passes the same rig at every gate, and a 7.3 MB prompt
pass crossed the WAN stream fine — size alone is not it. The stream path also
has no receipt acknowledgement, so a stalled frame costs the full segment
deadline (600 s) where rr's `FORWARD_ACK` fails over in seconds. Both need
answers — the stall's cause, and an ack or read deadline — before the default
flips.

**What a change must keep:** only the coordinator's setting matters (every node
registers the acceptor since v0.1.0-alpha.2); any send failure falls back to
request-response for that forward and evicts the stream; a stream read error
resolves the pending result with an error at once (faster than rr's ACK
deadline for a peer that went away).

## A speculative verify is walked where the logits are (2026-09-27)

**What it replaced.** Every speculative verify — the n-gram loop's hits AND
misses, and DSD's rounds — asked the last segment for `spec_logits`: one
full-vocabulary f32 vector per verified position. The coordinator then ran the
acceptance rule on them. On a 128K vocabulary that is 513 KB per position,
2.5 MB for a five-position round (measured on `split_rig.sh repeat` with a
v0.3.209 tail: 66 results of 513,070 bytes, 22 of 2,565,182). On a
Thailand↔Belgium link at 20-50 Mbit/s that transfer costs more than the round
trips speculation saves (`docs/plans/split_speculation.md` projects DSD as
shipped at 1.8-3.4 tok/s against 2.85 with no speculation at all).

**What it does now.** The last segment runs `sampling::sampled_accept_reject`
itself — sample each position with the caller's sampler, keep the drafts while
the sample agrees, stop at the first disagreement with the token it sampled
there (or the bonus after all of them) — and answers with those ids. It is
SpecExec's walk (arXiv 2406.02532); for a deterministic draft (an n-gram match,
a drafter's argmax) it IS the speculative-sampling rule, so the path is exact at
any temperature and DSD lost its greedy-only gate (the argmax-accepting
single-segment Item 2 keeps it).

**Measured** (`split_rig.sh repeat`, llama-3.2-3b, A=[shard 0], B=[1,2,3], both
on the new build): all 273 results B sent were token ids of 43-59 bytes (237 ×
one token, 12 × two, 9 × three, 15 × five), none a vocabulary. Replies scored
against llama.cpp: 119 of 121 tokens rank 1, worst rank 2 by 0.130 logits,
three runs identical. With B on v0.3.209 (the sender gate withholds the walk)
the reply was byte-identical, from logits walked on the coordinator.

**What a change must keep:**
- **Gated at the SENDER** on `SPEC_WALK_AT_TAIL | FORWARD_SAMPLING |
  FORWARD_GENERATED_IDS` (`pipeline::peer_walks_at_tail`). An older peer rebuilds
  the seal's AAD from the flags it parsed (`flags & 1`), so bit 1 would fail
  every encrypted verify it was sent. Our own worker always walks.
- **One writer of the flags byte**, `layer_forward::spec_trailer_flags`, for the
  plaintext frame, the encrypted frame and the AAD; an unwalked forward encodes
  byte for byte as before (`a_walk_at_the_tail_survives_the_wire_and_changes_nothing_when_unset`).
- **A walk travels with the caller's sampling or not at all** — the pool drops
  the flag when `sampling` is absent rather than walk at the worker's defaults.
- **`VerifyReply::accept` reads the shape the tail ANSWERED in**, not the one it
  was asked for, and refuses non-finite logits, too few rows, an empty walk, or
  a "kept" token that was never drafted. The old rule answered non-finite logits
  with `(empty, 0, false)` and callers emitted token 0 as the "bonus".

**From the rules file (moved 2026-10-02):**

A verify's last segment walks the drafts with the caller's sampler and answers
with token ids (`LayerForward::spec_walk_at_tail`, `0x03` flags bit 1, gated at
the SENDER on `features::SPEC_WALK_AT_TAIL`) — 43-59 bytes back where an older
tail sends 513 KB per position (2.5 MB for five on a 128K vocabulary).
**`sampling::sampled_accept_reject` is the one rule** and runs on either side;
**`pipeline::VerifyReply::accept` is the one place** a reply of either shape
becomes accepted tokens, and it refuses what no honest tail sends (non-finite
logits, a "kept" token never drafted). **`layer_forward::spec_trailer_flags` is
the one writer of the flags byte** for the plaintext frame, the encrypted frame
and the AAD. A walk travels only with the caller's sampling beside it.

→ `docs/invariants/network.md` § "A speculative verify is walked where the logits are"

**And it can walk with the request's SHARED noise** (`LayerForward::coupling_seed`,
the `0x0B` trailer, gated on `features::COUPLED_SAMPLING`): Gumbel-max keyed by
(seed, ABSOLUTE position, token id) — `inference::coupled_noise`. Row i of a
verify at `index_pos` predicts position `index_pos + 1 + i`; a drafter keys guess
k at `current_pos + 1 + k`. **Never a relative position** — the two sides then
draw different noise for the same token and agreement collapses silently
(`a_drafter_keyed_at_the_samplers_positions_is_accepted_and_one_off_is_not`).

## A check travels a chain like a decode step does (2026-10-02)

**What it replaced.** Decode steps had chained since 2026-08 (a run of remote
segments hands activations along; the tail answers), but a speculative CHECK
visited every segment through the coordinator: `forward_verify_through_segments`
looped segments one at a time. The n-gram loop — the DEFAULT split path — sends a
check every round, hits and misses alike, so a plan of N remote segments paid N
of the coordinator's round trips per round where a plain step paid one. The
delegated split (FUTURE_WORK #143) makes the head the coordinator, which removes
the requester's distance but not this: a delegate leading [itself, C, D] still
paid C and D separately.

**The mechanism.** The coordinator plans the run with the SAME `plan_chain` the
decode loop uses, filtered to hops advertising `features::CHAINED_VERIFY`, and
sends the head one forward carrying the chain, the reply-to (`0x07`) and the
TAIL's walk (computed for the answering node, not the head). Each hop's worker
computes its layers and ignores the check fields unless it is the model's last
segment (`model_worker::want_spec_output = spec_logits_requested && is_last`),
so the forward keeps them; `layer_forward` hands them on — guesses, walk flag
(gated on the next hop's `SPEC_WALK_AT_TAIL`), seed (`COUPLED_SAMPLING`), rewind,
history (`FORWARD_GENERATED_IDS`). A run that ends short of the last segment
returns hidden states and the loop resumes after it; one that reaches it returns
the walk. Unlike a decode chain, a check may chain with penalties set: its
history rides the head's forward and every hop hands it on.

**What it must keep.** The bit gates the COORDINATOR: an older hop would forward
activations only and the tail would answer a decode step — the wrong shape,
silently. A hop that finds the next one without the bit fails the run out loud
("the next segment cannot carry it") rather than forwarding it. The waiter pins
the tail and lists every hop in `chain_members` (any may refuse), and a chained
check passes `ResendOnRefusal::Never` — the refusal may be any hop's. A failed
chained check fails the request exactly as a failed star check did; nothing
re-runs it unchained mid-round (the earlier hops' caches already hold the
positions).

**Measured** (rig, 2026-10-02, `split_rig.sh remote REMOTE_NODES=3 DELAY_B=150`, A/B by
`SWARMLLM_CHAIN_VERIFY=0`, one binary): the n-gram request 2.03 → **2.66 tok/s**
(+31%); the mechanism counted — C chained 214 forwards with it on and 121 off, the
difference exactly that request's 18 hit + 75 miss rounds. The standard loop's
request is unchanged (2.57 vs 2.55): its decode steps were already chained.

**Tests**: `a_check_travels_a_chain_of_peers_that_can_carry_it` (chain + reply-to
+ the tail's fields on the head forward; controls: `may_chain = false`, and a tail
without the bit). Rig: `split_rig.sh remote` with `REMOTE_NODES=3`, `DELAY_B`, A/B
by `SWARMLLM_CHAIN_VERIFY=0`.

**In short** (the rule statement as it stood in `.claude/rules/arch-network.md`). `forward_verify_through_segments` sends a run of remote segments ONE forward
with the rest of the run as its `chain` — every hop advertising
`features::CHAINED_VERIFY` (gated at the coordinator: an older hop hands the tail
only activations, and it answers a plain decode step). A hop copies the check's
fields — guesses, walk flag, seed, rewind, history — onto the onward forward,
each gated on the next hop's own bit; a worker ignores them unless its segment is
LAST (`want_spec_output`). The waiter pins the tail and admits every hop's
refusal; a chained check resends on nothing (`ResendOnRefusal::Never`).
`PipelineExecutor::verify_may_chain` decides; `SWARMLLM_CHAIN_VERIFY=0` is the
control arm.

## A connection the swarm denied is forgotten (2026-10-02)

**Found** by the delegated split (FUTURE_WORK #143): its reply streams back one
request-response send per token, and the n-gram loop emits several tokens per
round, so sends leave in bursts. The first delegated request failed on every rig
run with resend asks for the SAME token ids (19-20, 31-32, …): deterministic.
Per-token DIAG showed the serving node queued them and the requester's network
layer never saw them; `-vv` plus a log in `try_send_request` showed the serving
node's request-response listing five connections to the requester where the swarm
had three (`connection limits configured max_per_peer=3`). Requests 43 and 44 —
the fourth and fifth of a five-token burst — went to ConnectionIds 34 and 35,
which had no substream, no response and no failure.

**Mechanism.** `handle_established_inbound_connection` /
`..._outbound_connection` build a handler and `preload_new_handler` pushes the
connection onto `connected`. libp2p-swarm 0.47's own docs: "when any composed
behaviour returns an error the connection will be closed and a
[`FromSwarm::ListenFailure`] / [`FromSwarm::DialFailure`] event will be emitted"
— carrying the `ConnectionId`. `connection_limits` is composed after
request-response, so a connection over the cap is handed to request-response and
then denied; upstream request-response clears only `pending_outbound_requests` on
a dial failure and ignores a listen failure, so the entry stays. A ghost has no
handler and nothing pending, and `connection_rank` ranks fewest-unanswered FIRST:
the moment each real connection has one request in flight, the next send goes to
a ghost. A capped peer is redialled every ~5 s (seen on the rig), so ghosts
accrue for as long as a node runs.

**Fix.** `forget_denied_connection(peer, connection_id)` on `ListenFailure` and
`DialFailure`: remove the entry (the peer's too when it was the last) and emit
`OutboundFailure::ConnectionClosed` for each request it was handed, which the
network manager reports or resends. Test:
`a_connection_the_swarm_denied_is_forgotten_and_its_request_fails` — a two-request
burst reaches the connection about to be denied (fixture), the denial forgets it,
its request is reported failed, the next send goes to the real connection; it
fails with the `ListenFailure` arm removed. Rig after: 0 resend asks where every
earlier run had 3-4.

**What it explains, retrospectively** (not re-measured): the 2026-08-05 warning
patched into `on_connection_closed` ("swarm reports no connections left but this
behaviour still held some", right after PEX dialled four peers); gotcha #353's
"newest connection dead in one direction" that swallowed a first send; and the
"libp2p rr can silently drop sends under load" this file's ACK-timeout section
was built to survive. Those defences stay — they bound any OTHER silent loss.

## A substream sends with its protocol proposal (V1Lazy); a ping sample is a cost, a distance is converted (2026-09-27)

**What it replaced.** libp2p negotiates every substream with multistream-select;
under `Version::V1` the dialer waits for the listener's confirmation — "always at
least one dedicated round-trip message exchange before application data"
(multistream-select 0.13.0 docs) — and request-response opens a substream per
message. So every token of a split request paid one extra round trip on each
leg. `Version::V1Lazy` sends the first message with the proposal when one
protocol is offered; it is wire-identical for the listener, so an older peer needs
nothing. Substrate runs it for every substream and measured request answer time
halved (paritytech/substrate#7606). libp2p-swarm 0.47 has no per-substream
setting, so it is the swarm-wide `with_substream_upgrade_protocol_override`;
protocols offering several versions (gossipsub) still negotiate in full.

**Measured, real link** — A/B/A in one binary (`SWARMLLM_SUBSTREAM_V1=1` = V1),
qwen2.5-coder-7b split this node (RTX 3070, TH) L0-14 → bf7b (RTX 4050, BE,
v0.3.210) L14-28, request-response path, n-gram loop off, 64 tokens × 3 per arm:
V1Lazy **1.67 tok/s** (598 ms/token) → V1 **1.23** (811) → V1Lazy **1.76** (568).
Mechanism: the PEX ping to the same peer read 248 ms lazy against 497 under V1
(ICMP 206). Only OUR dialer changed; the peer's result messages still paid V1,
so a peer on this build should take a further round trip off.

**The trap it opened (gotcha #742).** The same halving reaches every
request-response SAMPLE, `PeerInfo::latency_ms` included — the figure #356's
addendum recorded as ~2×RTT and kept. Cost readers (routing's per-hop price,
`DELEGATE_MAX_LATENCY_MS`, ACK deadlines) should move with it: every exchange they
price is that much cheaper. Two readers mean DISTANCE and must not: the LAN
heuristic (`rtt_ms < 5 → is_lan_peer`, sticky, and private mode admits LAN peers
by default) and the Vivaldi coordinate. Under V1Lazy the same 4 ms ping is one
4 ms round trip — across town — not two 2 ms ones.

**What a change must keep:**
- `network::manager::physical_rtt_ms` is the ONE conversion from an exchange
  sample to a round trip (÷ the round trips the negotiation costs); the LAN check
  (`exchange_says_lan`, `LAN_PHYSICAL_RTT_MS = 2.5`, the distance the old rule
  meant) and `observe_network_coord` go through it. Test:
  `a_lan_boundary_means_the_same_distance_whatever_the_negotiation`.
- Never compare a raw request-response sample to a distance constant.
- The negotiation mode is read ONCE (`substream_negotiation`, a `OnceLock`) so the
  swarm and the conversion cannot disagree within a process.

## Shared noise for a speculative walk (2026-09-27)

**What it adds.** A walk accepts a FIXED draft with probability p(draft), which at
temperature > 0 is well below how often a close drafter agrees with the target:
for a 3-bit copy of Qwen2.5-Coder-7B's far half, 92.5% coupled vs 71.3% fixed at
T=1.0 (`docs/plans/split_speculation.md` Phase 4). With the request's shared noise
(`inference::coupled_noise`: Philox4x64-10 keyed by seed, absolute position and
token id; Gumbel-max over each side's own filtered logits) each side still draws
an exact sample of its own distribution, and a close drafter draws the same token
(Daliri et al., arXiv 2408.07978).

**Measured** (`split_rig.sh repeat`, llama-3.2-3b, DSD with a 3-bit far-half
shadow as its drafter, γ=4, `RIG_TEMPERATURE=0.7`, n-gram lookup off, 3 requests
per arm, one binary, `SWARMLLM_SPEC_COUPLING=0` for the fixed arm): guesses
accepted **80.6% vs 70.0%**, tokens per round trip **4.11 vs 3.71**. With DSD's
n-gram cascade tried first, the coupled arm fell to 3.47 — an n-gram guess is a
fixed token — so DSD skips it whenever shared noise is on.

**What a change must keep:**
- Positions are ABSOLUTE: row i of a verify at `index_pos` samples position
  `index_pos + 1 + i` (worker `walk_verified_positions`, coordinator
  `VerifyReply::accept`), and DSD's `DraftPick::first` is `current_pos + 1`.
- The seed goes only to a peer advertising `COUPLED_SAMPLING`; the pool keeps it
  only beside a walk with the caller's sampling. A tail without it walks with its
  own draw — still exact, just in agreement less often.
- `0x0B` is written by `layer_forward::append_coupling_trailer` for the plaintext
  frame, the encrypted frame and the AAD alike; absent, a frame is byte-identical.
- Exactness is pinned by `a_coupled_sample_follows_the_ordinary_samplers_distribution`
  (with its control) and pruning by `pruning_never_changes_the_coupled_pick`.

## A stream of verifies runs in its order, and each answer names its number (2026-09-30)

**What it is for.** Split speculation's rounds are strictly serial — draft, run
the near half, send, wait a round trip for the far half's verdict, draft again —
so the drafting and the near half sit inside every round trip. A stream keeps
several verify chunks of ONE request in flight to the segment that samples, each
built on the guess that the chunks before it will be kept (`docs/plans/split_speculation.md`
§ 4b). PipeInfer (arXiv 2407.11798, SC'24) runs the same scheme across MPI
nodes; its two requirements carry over: runs are executed in the order sent
(MPI's non-overtaking rule for one sender, receiver and tag), and micro-batches
of 1-4 tokens beat large ones once several are in flight.

**What assumed one forward per request, and what each became:**
- **The serving node ran forwards as they arrived.** An encrypted forward is
  opened in its own task and the dispatcher spawns a handler per forward, so two
  a few milliseconds apart race to the worker — which writes each at its cache's
  CURRENT length while RoPE rotates by `index_pos`: swapped chunks are silently
  read into each other's positions. Now `LayerForward::stream_seq` (the `0x0C`
  trailer, sealed in the AAD) numbers a stream's forwards, and
  `daemon::state::forward_streams` makes forward N wait until N-1 has ended on
  this node (`STREAM_TURN_WAIT`, 60 s, then refused). The number's high 12 bits
  name the ATTEMPT (`types::inference::stream_seq`): a router retry reuses the
  request id, and numbered from 0 per request the retry's chunk N and the dead
  attempt's chunk N would share a number — one's answer, or one's late refusal,
  would land on the other's wait (gotcha #749). The first build restarted the
  numbering on a prompt pass instead; that closed the serving side and left the
  coordinator's waits exposed. The worker checks the result: a streamed forward
  whose `index_pos` is not the cache's length after truncation is refused.
- **A restart skips what it supersedes.** A streamed forward carrying
  `truncate_kv_to` restarts its stream — the coordinator sends one only after a
  refused guess — so every earlier turn not yet run on the serving node is
  answered unrun (`TurnRefused::Skipped`), still in turn order so no two ever run
  at the worker at once. PipeInfer's early cancellation, signalled by the restart.
  On the shared-card rig it took the stream from 17.8 to 25-27 tok/s. A stale
  chunk the node has already STARTED is not stopped: running the restart beside
  it would route one's reply to the other (gotcha #180).
- **The worker's reply routing is keyed by request id** (gotcha #180). The gate
  keeps one forward of a stream at the worker at a time, so that map needs no
  change.
- **The cancel registry held one forward per request**, so a second
  overwrote the first and the first, finishing, withdrew the second — which no
  `CancelInference` could then reach. It holds every forward of a request now
  (`inbound_forward_aborts`, `two_forwards_of_one_request_are_each_cancellable`).
- **The coordinator's waiters were keyed by request id.** `pending_layer_results`
  is keyed by `WaiterKey` (request + stream number) now, and a streamed answer
  echoes its number (`ResultStep::stream_seq`, the `0x08` result trailer, after
  `0x07` so an older decoder reads the step and stops). Position and range alone
  cannot name a chunk: a stream restarted after a refused guess sends its next
  chunk at the position a discarded chunk also started at.
- **A failure this node manufactures names no step** (a send that failed, a
  receipt that never came). It is about the LINK to one node, so with no
  unstreamed wait on the request it ends the stream's OLDEST wait pinned to that
  node — the one its coordinator reads — rather than leaving every chunk to sit
  out its deadline. Gotcha #229 is why it is pinned: a failure about node X must
  never end a wait on Y.
- **The per-peer cap counted each chunk as its own work, and a chunk it refused
  left a hole** (found by the v0.3.215 release gate, 2026-10-01, step 12e
  `failover` with speculation on: A on the card, B on the processor). B served
  the healthy request's stream until five of its chunks were in flight at once —
  one running, live ones, and ones a restart had superseded still waiting to be
  skipped — and refused the fifth at `max_forwards_per_peer`'s floor of 4. That
  refusal never reached `handle_layer_forward`, so the chunk never took its turn;
  the coordinator, meanwhile, had restarted past it. The restart and the three
  chunks after it waited `STREAM_TURN_WAIT` for that turn (B: "streamed check 21
  waited 60s for the check before it, which never came", ×4), the coordinator's
  52 s segment deadline fired first, and the request re-planned away from a
  healthy B — 200, about a minute late, with the takeover the step exists to
  watch never happening ("the reply finished before B started computing").
  Reproduced on purpose the same morning on the plain two-node shape (A card
  0-3, B processor 4-7, window 6 > the cap): .215 refused 19 chunks, 10 waited
  the 60 s, all 3 requests timed out — and with no other holder to re-plan onto,
  each ended with its first ~10 tokens, salvaged and marked `error`. The fixed
  build on the same rig: 0 refusals, 0 waits, 0 retries, 120 tokens each (window
  6 and 3). A restart skips only turns that ARRIVE, so nothing could close the hole.
  Two changes, each sufficient for this instance and both needed in general:
  **a stream is one piece of its sender's work** — `dispatch::StreamWorkSlot`
  gives a stream's chunks one shared per-peer slot while any is in flight, up to
  `MAX_STREAM_CHUNKS_HERE` (twice the coordinator's largest window), the way
  HTTP/2's `SETTINGS_MAX_CONCURRENT_STREAMS` counts streams and leaves a stream's
  frames to its own window (RFC 9113 §5.1.2, §6.9) — and one node-wide permit,
  since queued chunks compute nothing (two busy streams would otherwise fill a
  `Minimal` node's 8). This is per-STREAM admission without the end-of-request
  signal #123 says per-REQUEST admission lacks: the hold lives exactly while a
  chunk is in flight, so its own count is the signal; and **a refused chunk is
  stepped over** — `refuse_forward` calls `ForwardStreams::refused_on_arrival`,
  and the stream skips that number when it comes due, as it does a chunk that
  took its turn and failed in the handler. TCP closes a receiver's hole by the
  sender's retransmission; nobody resends a refused check, so the refusing node
  closes it. The coordinator was answered either way: it ends the stream at that
  answer, or a restart it already sent cuts both caches back past the hole.
  Gated by a feature bit with no wire change (`features::STREAM_AS_ONE_WORK`):
  v0.3.213-v0.3.215 still count each chunk, so a coordinator streams only to a
  node advertising it and gives an older one rounds.

**What a change must keep:**
- Streamed forwards go only to a peer advertising `features::STREAMED_VERIFY`
  AND `features::STREAM_AS_ONE_WORK`. An older peer would fail the seal on
  `0x0C` and would also run the chunks concurrently; one with the first bit only
  refuses a busy stream's chunks and stalls on the hole.
- The turn is given back by DROPPING it (`forward_streams::Turn`), on every
  exit of `handle_layer_forward` — an early refusal or an abort included —
  or every later chunk waits out `STREAM_TURN_WAIT`. **And a forward refused
  BEFORE the handler** (the dispatcher's admission caps) gives up its number
  through `refuse_forward`; a new refusal point ahead of the handler must go
  through it too. Pinned by
  `a_restart_does_not_wait_for_a_turn_refused_on_arrival` (the gate's shape) and
  `a_streams_chunks_share_one_per_peer_slot`.
- Pinned by `forwards_run_in_their_streams_order_whatever_order_they_arrive_in`,
  `a_streamed_answer_reaches_the_wait_its_number_names` (red with either the
  number's routing or the link-failure rule toggled off, 2026-09-30) and the
  codec tests `a_stream_number_*` / `a_streamed_answer_names_its_number_after_the_step`.

**From the rules file (moved 2026-10-02):**

A streamed check keeps several verify forwards of ONE request in flight
(`LayerForward::stream_seq`, `0x0C`, gated on `features::STREAMED_VERIFY`).
**`daemon::state::forward_streams` runs them in number order** — forward N waits
until N-1 has ended here, since the worker writes at its cache's length and
nothing else orders them; the turn is given back by DROPPING it, on every exit.
The number's high bits name the ATTEMPT (`types::inference::stream_seq`, #749),
and a forward carrying `truncate_kv_to` restarts the stream: earlier turns not
yet run are skipped, in order — never run beside another (#180).
**`pending_layer_results` is keyed by `WaiterKey`** (request + number) and an
answer echoes its number (`ResultStep::stream_seq`, `0x08`): position alone
cannot name a chunk once a restarted stream reuses one. A failure this node
manufactures (no step) ends the stream's OLDEST wait on that node only (#229).
The cancel registry holds every forward of a request.
**A stream is ONE piece of its sender's work, and a chunk refused on arrival is
stepped over** (2026-10-01): its chunks share one per-peer slot and one node-wide
permit (`dispatch::StreamWorkSlot`, up to `MAX_STREAM_CHUNKS_HERE`), and every admission
refusal goes through `refuse_forward`, which gives up the chunk's number
(`ForwardStreams::refused_on_arrival`). Counted one each, a stream with superseded
chunks queued was refused at the per-peer cap of 4, and every chunk after the
hole waited 60 s — a restart skips only turns that ARRIVE. A coordinator streams
only to a peer advertising `features::STREAM_AS_ONE_WORK`.

→ `docs/invariants/network.md` § "A stream of verifies runs in its order"

## One upload per model id, on every node (2026-10-02, FUTURE_WORK #151)

**What it replaced.** A model id is derived from a GGUF's FILE NAME
(`model::canonical::model_id_for_gguf_filename`), so independent uploads of one
model and quantisation share an id — `bartowski/…/Qwen2.5-Coder-7B-Instruct-
Q4_K_M.gguf`, `Qwen/…/qwen2.5-coder-7b-instruct-q4_k_m.gguf` and a third-party
requant were all `qwen2.5-coder-7b-instruct-q4-k-m`, within 800 bytes and
sharing no part hash (gotcha #406). Every part was verified — against the upload
its own node fetched. Nothing made two nodes fetch the SAME upload: `hf_sources`
had six writers with six rules (the first `HfSourceGossip` heard, the dashboard
click, the first HuggingFace search hit — ordered by downloads, so it moved —,
a `hf_source.json` on disk, the auto-manage discovery). Measured 2026-10-01:
**9 of 20 models with peer holders had holders of another upload**
(`peers_other_build`), and `shard_holders` (correctly) refuses to split across
them. Three paths also MIXED uploads on one node: the auto-manage HuggingFace
download fetched part *i* from `hf_sources` and wrote its hash into whatever
manifest was registered; the dashboard download merged into an existing
directory named by the caller's `model_id`; four header fetches took
`gguf_header.bin` — the file that says where every tensor is — from
`hf_sources` beside parts a peer had supplied.

**The rule** (`model::canonical`): uploads of one model are totally ordered —
a pinned reference model first (`model::reference`), then the publisher's
position in `TRUSTED_HF_PUBLISHERS` (official authors, then curators), then
anyone, ties by name — and every node uses the best upload anyone has claimed.
Claims only grow by gossip and the choice is the maximum under a fixed order,
so it converges: a state-based max-register CRDT (Shapiro et al., 2011 — merge
is "keep the better", commutative, associative, idempotent). Petals keys a
model's swarm identity by its HuggingFace repo and Ollama resolves `name:tag` to
one manifest digest; this keeps the friendly id and makes it resolve to one file.

**Verified anonymously.** A candidate is adopted only after a probe WITHOUT the
node's HuggingFace token (`probe_public_upload`): a node with a token can read
gated or private repos the rest of the swarm cannot, and an upload chosen on its
word would be one nobody else could fetch. A refused upload is skipped for 24 h
(not found / private) or 30 min (transient), so the next-best is chosen.

**Healed, not just prevented** (`model::auto_manage::canonical`, a background
task that runs whether or not auto-manage is on — repair, like
`complete_pending_shard_fetches`):
- A node holding NONE of a model it may fetch registers the canonical upload's
  manifest (from its header) and fetches against it; the dispatcher drops a
  peer's manifest of another upload once the choice is known
  (`manifest_is_another_upload`); `canonical_allows_acquisition` holds every
  download path until the choice is made (`trigger_download`, the scorer).
- A node holding parts of the same SHAPE has each held part's first 64 KB
  compared with the upload's bytes on HuggingFace (`parts_are_from`) — uploads
  differ in tensor bytes everywhere, so this is decisive and costs 64 KB a part —
  and its header compared by BLAKE3 with the upload's; a wrong header (a
  peer-provisioned node's) is replaced with the upload's header and side files,
  and the model reloaded.
- A node holding another upload fetches the canonical parts covering its layers
  into `<data_dir>/canonical/<model>` by byte range (never a whole GGUF), keeps
  serving its old parts meanwhile, and swaps when every part is in and the model
  is idle: old parts, header, manifest and side files deleted, new ones moved in,
  origin hashes of the old upload forgotten (`forget_origin_verified_for_model`
  — kept, they would refuse the new manifest as "a different build"), holders
  re-announced complete-for-model, reloaded. One switch at a time.
- `hf_sources` is written ONLY by `SharedState::write_hf_source` behind
  `note_origin_claim` / `adopt_canonical_build` (and the startup restore of the
  verified choice), and a header is fetched only by `fetch_model_header`, which
  refuses a source whose file is not the size the manifest says. Guards:
  `a_models_source_is_written_only_through_the_canonical_choice`,
  `a_models_header_is_fetched_only_from_the_upload_its_parts_are` (both with
  planted-violation null controls run 2026-10-02).

**What a change must keep:**
- The ORDER of `TRUSTED_HF_PUBLISHERS` is part of the wire contract: two
  versions that rank two uploads differently move the swarm back and forth.
  Never reorder; inserting a publisher is safe.
- Every canonical layout is cut at the DEFAULT shard size
  (`canonical_shard_size_bytes`), never at a node's `model.shard_size_mb`.
- A claim must name the model (`origin_names_model`): a file whose name gives
  another id is never a candidate, or one gossip message could move a model's
  holders onto any file.
- Offline mode keeps the old behaviour (it never reaches HuggingFace), and a
  node every candidate of which HuggingFace refuses fetches as before rather
  than never.

**Measured on a 2-node rig (2026-10-02):** A downloaded hugging-quants' Llama-3.2-1B
Q8_0, B bartowski's (same id). A first adopted its own upload (the only claim it had
heard), then bartowski's two minutes later when B's claim arrived — the max-register
in action — and switched in 27 s; 157 s from the downloads starting to both reading
`canonical` with `peers_other_build` 0. Every part, the header and both side files
came out byte-identical, and a 3-segment split across the two answered. The run also
caught a pass replacing the manifest while A's own dashboard download was between its
header and first part — `settle` now waits while a download of the model is running.

**The .220 gate (step 12k) found a fourth place two uploads can meet: the
manifest ON DISK.** A dashboard download of upload X had its registry manifest
replaced by a peer's manifest of Y before any part arrived (no origin knowledge
yet, last-writer-wins), then wrote X's manifest to disk; X's and Y's Q8_0 tensor
bytes were identical, so the part check passed and the header was replaced —
and the worker, loading the disk manifest, read every tensor 3,808 bytes off
("position … is in a missing region (total_size=<X's size>)"). `ensure_manifest`
now makes the registry AND the disk manifest describe the canonical upload
(rebuilt from its header when not), and `CanonicalBuild::describes` also compares
each part's first-tensor offset where a manifest carries its table. Verified on
the same race: "Rewrote this model's manifest", then "Paris" whole and split.

**Known limits (FUTURE_WORK #151):** a switch fetches from HuggingFace, not from
canonical holders over P2P; a coordinator holding none of a model routes on a
manifest with placeholder hashes until a holder's gossip fills them, so for that
window `shard_holders` cannot exclude another upload's holders; a node in
offline mode never switches.

**From the rules file (moved 2026-10-02):**

**`model::canonical` is the single answer to "which file IS this model"**: uploads
of one model id are totally ordered (reference pin, then `TRUSTED_HF_PUBLISHERS`
position, then name) and every node uses the best one anyone has claimed, after
checking it on HuggingFace WITHOUT its token. `hf_sources` is written only through
`note_origin_claim` / `adopt_canonical_build`; a header only through
`fetch_model_header`; every download path asks `canonical_allows_acquisition`; a
peer's manifest of another upload is dropped. `auto_manage::canonical` heals a node
holding another upload — stage the canonical parts, swap when idle — and replaces a
header from another upload. ⚠ **Never reorder `TRUSTED_HF_PUBLISHERS`** — its order is
a swarm-wide contract. Nine of twenty models were held as two or three uploads that
could never be split together (#151).

→ `docs/invariants/network.md` § "One upload per model id"

## A key derived while ANSWERING an exchange waits until the peer proves it has it

A rekey is two messages and the second can be lost, so a responder that installs
before answering can be left holding a key the initiator never saw — one
direction dead, neither end able to notice. `accept_ephemeral_exchange` parks it
in `unconfirmed` and keeps sealing with the key both ends still agree on; `open`
promotes it, carrying its replay window, on the first message that opens under
it. The initiator seals `SESSION_CONFIRM_MARKER` as it installs —
`complete_ephemeral_session` RETURNS those bytes, so the seal and the install
cannot come apart — and ordinary traffic confirms just as well.

**Gated on `features::SESSION_KEY_CONFIRM`**: a peer that cannot confirm is
answered the old way, or the failure is only mirrored. Answering an exchange also
drops our own outstanding initiation, so two crossing rotations cannot leave each
end sealing with a key the other holds only as superseded. WireGuard's rule, for
the same reason.

→ `docs/invariants/network.md`

## A tensor forward is acknowledged on receipt; a result is always its own request (2026-08-21)

`requests.rs` answers an inbound `LayerForward` with `SwarmResponse::Ack` the
moment it decodes; `tensors::handle_send_tensor_result` always sends the result
via `send_tensor_result_as_request`. There is no held `ResponseChannel` any
more — the "single substream per token" path and its map are gone.

Why: with the request held open until the result was ready, a coordinator
could not tell "computing" from "never received it", and a forward that landed
on a peer that had gone quiet cost the WHOLE segment deadline (300 s, twice in
one request on the live swarm). Now the serving node advertises
`features::FORWARD_ACK`; the coordinator's stale sweep fails an unacknowledged
forward to such a peer at `forward_ack_deadline_secs` (RTT-scaled, 10-90 s) so
the pipeline fails over in seconds. **The compute deadline stays with the
pipeline** — the sweep never reaps a slow answer, only a missing receipt; the
comment at the sweep recounts how an earlier reaper synthesised timeouts for
exactly the slow peers the measured deadlines were built for.

Rules: do not gate the fast-fail on anything but the peer's `FORWARD_ACK` bit
(an older server answers only with the result, which may take minutes); the
ACK must be sent BEFORE any work, from the network manager, so no dispatch
path can forget it; and both halves are compatible with older peers in both
directions — an old coordinator accepts an ACK response and a result arriving
as a request, an old server is never held to the ACK deadline. This also
retired the chain-specific addressing branches from earlier the same day
(gotcha #354): one rule now covers chained and unchained results.

## "Is this connection direct?" — `network::relay::addr_is_direct_transport`

The single answer, for BOTH layers that choose a connection. A relay-carried
connection looks different from each end: the dialer sees
`/…/p2p-circuit/p2p/<peer>`; the LISTENER sees a bare `/p2p/<peer>` — no
transport hop at all. `is_relay_circuit_addr` catches only the first form, and
the vendored request-response layer recorded NO address for inbound connections
(upstream behaviour), classifying `None` as direct. So a relay-carried inbound
connection was "direct", and — being the newest with nothing pending — it won
every send: two LAN nodes 0.7 ms apart measured 60-800 ms to each other and
forwarded every tensor through the anchor in Europe (gotcha #356, 2026-08-21;
gotcha #179 had fixed the classifier but inbound never had an address to classify).

Rules that follow: the vendored inbound handler records the send-back address
(`vendor/libp2p-request-response/src/lib.rs`, SwarmLLM patch); the manager's
`peer_direct_conns` is gated on `addr_is_direct_transport` (a real `/ip4`,
`/ip6` or `/dns*` hop AND no circuit), not on the circuit check alone. Any new
"prefer direct" logic uses the same predicate. `latency_ms` in the peer
registry is measured over whichever connection the rr layer picks, so a wrong
pick here is not a cosmetic error — it is added to every number routing uses.

The loop-stall tripwire in `NetworkManager::run` (per-arm, ≥100 ms → `DIAG:
network event loop stalled`) exists because this investigation's first
hypothesis was the loop; a shadow node cleared it in four minutes. Keep it.

## Centralised Wire-Format Helpers

These helpers exist as the single source of truth for invariants that
silently break at the wire if duplicated:

- **`network::protocol::build_layer_forward_aad`** — encryption AAD
  bytes for `LayerForward` envelopes. ⚠ **A new trailer is NOT a no-op for an
  older peer.** Decoders reconstruct the AAD from the trailers they PARSED, so
  one they do not know makes every encrypted forward fail to open — every
  optional trailer must be feature-gated at the sender, not merely "ignored" at
  the receiver. `0x08` (the decoded-so-far ids) is the worked example. Both encrypt
  (`network/manager/tensors.rs`, `network/pipeline_stream.rs`) and
  decrypt (`decode_layer_forward_encrypted`) MUST go through it.
  Adding a new authenticated field to `LayerForward` means extending
  this helper, not appending bytes on the encrypt side. **Every optional
  trailer the wire carries must be bound here**, and `tp_meta` (0x02) was not
  until 2026-09-14 — it decides which SLICE of a tensor-parallel layer the
  receiver computes (`tp_rank`/`tp_size` key the worker's SplitModel cache and
  drive `pre_split_for_tp`), rides in cleartext after the sealed payload, and a
  RELAY node forwarding tensor payloads for others is a legitimate endpoint that
  can flip it. The failure is a silently WRONG AllReduce contribution, not a
  rejected request. Extending the AAD only affects forwards that carry the
  trailer, which is the compatibility argument each bump rests on.
  ⚠ **A trailer a receiver CLAMPS is clamped AFTER the AAD is rebuilt**, from the
  bytes as sent — `0x0A`, the caller's sampling (gated on `FORWARD_SAMPLING`,
  → `docs/invariants/network.md` § "The caller's sampling reaches a remote
  sampler"), is the worked example; clamping first fails the seal for any
  out-of-range value and reads as a key problem. Post-R100,
  the helper covers the cleartext header AND the spec/kv-truncate
  trailer fields; the decoder reconstructs AAD via the helper after
  parsing trailers (since trailer bytes don't appear contiguously
  on the wire — sealed payload sits between header and trailers).
  Post-R139, also covers the chunk-meta trailer (0x05) so chunked
  STREAM frames can't be reordered / truncated / substituted across
  transfers without Poly1305 rejection.
- **`network::pipeline_stream::chunk_layer_forward`** (R139) — splits
  a `LayerForward` at byte-offset boundaries into K chunks for
  STREAM-style chunked send. Returns the input verbatim wrapped in a
  single-element Vec when `activations.len() ≤ chunk_size_bytes`
  (single-chunk implicit fallthrough — no chunk_meta on the wire).
  Sender call sites that opt into chunked send MUST go through this
  helper rather than re-implementing the split; the chunk_meta
  values it sets are the contract the receiver's
  `try_assemble_chunked_forward` and `build_layer_forward_aad` both
  rely on.
- **`SharedState::resolve_pending_layer_result`** — the ONLY way to deliver a
  `LayerResult` into `pending_layer_results`. Never `remove(&request_id)` +
  `tx.send(...)` from a network path. The map is keyed by `request_id`, but a
  request that has failed over has TWO forwards outstanding: the abandoned one
  and the standby's. Resolving by id alone lets the abandoned forward's late
  error (from `fail_tensor_forward`, `fail_pending_forward`, or the
  stale-forward sweep) consume the standby's waiter — which then discards the
  standby's genuine result and surfaces the empty payload downstream as
  `Internal: Tensor bytes too short`. Observed live 2026-08-01: a request that
  would have completed in ~10s via failover failed after 181s (gotcha #229).
  Waiters record the node they expect in `PendingLayerResult::awaiting`; the
  helper checks and takes in one atomic `remove_if`. A bare `remove` is only
  legitimate for owner-side cleanup — a coordinator dropping its OWN waiter on
  an error path, or the health monitor's stale sweep.
- **`daemon::dispatch::timestamp_fresh_one_sided`** — generic
  one-sided staleness check (R94). Time units must be consistent
  across `ts`/`now`/`max_age`/`skew`. Use directly for any new
  timestamp gate; gossip and pre-signed-message helpers below are
  thin wrappers around it.
- **`daemon::dispatch::gossip_timestamp_fresh`** — private `u64`-ms
  wrapper used inside `daemon/dispatch/mod.rs` itself for the four
  inbound gossip handlers (`RegionShardSummary`, `ModelDemandGossip`,
  `WishlistAnnouncement`, `PoolModelAvailability`). The
  `network::manager::events.rs` GossipSub pre-filter uses
  `timestamp_fresh_one_sided` directly via an inline closure — both
  sites share the same one-sided invariant, but via the underlying
  primitive rather than this wrapper.
- **`credit::ledger::check_signed_freshness`** — one-sided staleness
  check for `chrono::DateTime<Utc>`-typed signed messages (balance
  reports, credit transactions, pool removals). Constants
  `CLOCK_SKEW_TOLERANCE_SECS` / `BALANCE_REPORT_MAX_AGE_SECS` are
  `pub(crate)` so all callers share the same window (gotcha #32). R94
  routed `pool/manager::handle_inbound_removal` through here.
- **`pipeline::pack_verify_tokens_to_le_bytes`** (R93) — packs `&[u32]`
  speculative-verify tokens as i64-LE bytes for the worker's
  multi-token decode branch. Shared by `speculative.rs::send_verify_batch`
  and `dsd.rs::forward_verify_through_segments`.
- **`pipeline::build_spec_verify_forward`** (R93) — constructs the
  18-field `LayerForward` envelope for spec verify (R139 added the
  18th field, `chunk_meta`). Adding a new field
  to `LayerForward` extends this helper, not the call sites.
- **`pipeline::build_kv_truncate_forward`** (R95) — sibling helper
  for stop-sequence KV-truncate signals (empty activations,
  `spec_logits_requested: false`).
- **`pipeline::register_pending_layer_result`** (R93) — cap-check +
  oneshot insert + `PendingLayerResultGuard` RAII (gotcha #45). Used
  by speculative prefill, speculative verify, and DSD verify;
  `distributed.rs` keeps two inline call sites that need `&mut self`
  or skip the cap during failover.
- **`storage::Database::with_write_table`** (R96) — opens a write
  transaction, runs a closure on the data table, commits on `Ok` or
  rolls back on `Err`. Used by `put_json`, `insert_raw`, `remove`,
  `clear_tree`, `replace_tree`. Read-side dedup deferred (lifetime
  constraints on `ReadOnlyTable`).
- **`swarmllm_types::ShardResponse::empty()`** (R97) — canonical
  empty/error response for refused requests, queue-full rejections,
  and disk read/seek/open failures. 8+ rejection sites across
  `network/manager/{requests,shard_transfer}` go through it.
- **`swarmllm_types::LayerResult::error(request_id, reason)`** (R106)
  — canonical empty/error LayerResult for failed pipeline forwards.
  Five rejection sites (`network/manager/{tensors,requests,mod}.rs`,
  `network/pipeline_stream.rs`, `daemon/dispatch/layer_forward.rs`)
  go through it. Adding a new field to `LayerResult` only requires
  updating this constructor — mirrors `ShardResponse::empty()`.
- **`network/manager/connections::try_enqueue_redial`** (R97) —
  dedup + cap + push for `pending_redial`. Used by both the
  active-pipeline and unregistered-peer reconnect paths.
- **`responses::types::raw_tool_kind_or_unknown`** (R93) — extracts
  the `type` field from a `ToolDef::Raw` JSON value with `<unknown>`
  fallback. Used by both Chat and Anthropic `translate_tools` error
  arms.
- **`cli::bail_if_no_api_key` / `cli::exit_daemon_unreachable`** (R96)
  — the canonical "daemon not running" / "daemon unreachable"
  messages. Used by `cli::{bench, chat, peers, status}`.
- **`model::auto_manage::spawn_check_and_load`** — canonical
  "shard landed → reload model → refresh dashboard" spawn. Always
  performs the three steps together: compute_vram_budget →
  check_and_load_model → signal_dashboard(ModelsChanged). Used by
  `api/admin_models/shards.rs::delete_shard`,
  `network/manager/requests.rs` shard-download landing, and
  `model/acquisition.rs::register_model`. New paths that complete a
  shard or shard-set acquisition MUST go through this helper rather
  than open-coding the three-step sequence.
- **`pool::invite::{encode_invite_code, decode_invite_code}`** (R140) —
  canonical `swarmpool://` v2 invite code codec. Encode JSON-serializes
  `InviteCodePayload` → ChaCha20-Poly1305 seals with a random embedded
  key → base64url; decode reverses with version + expiry + token-length
  validation. The decoder normalizes ANY user-pasted error to
  `SwarmError::Validation` (clean UX message) rather than `Internal` —
  the most likely failure cause is a truncated/mistyped paste, not a
  daemon bug. New entry points that accept v2 codes (CLI, MCP tool, web
  API) MUST go through `decode_invite_code` rather than parsing the
  blob manually. Adding a field to `InviteCodePayload` requires bumping
  `INVITE_VERSION` AND updating the decoder's mismatch error to point
  users at a daemon upgrade. `pool::invite::looks_like_v2` is the
  prefix-sniff helper used by API + frontend to route between v2 and
  the legacy 8-char path.

## Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer

Three rules, all paid for on 2026-09-21 by measuring a node that held no models
and served nothing while spending ~2.6 Mbit/s.

- **Publish on CHANGE, never per item per tick.** `broadcast_region_summary`
  emitted one message per known model plus one per `(model, region)` demand
  entry every 30 s, unconditionally. `swarm/regions` carried **87 messages a
  second inbound at 557 bytes each** — a message-RATE problem, which no payload
  shrink would have touched. Change-gate on a digest of what is being
  ASSERTED, and **exclude any timestamp**: fold the clock in and every message
  reads as changed, so the gate suppresses nothing while looking right in
  review.
- **Gossip only what this node MEASURED.** `region_demand` is written by the
  inbound `ModelDemandGossip` handler, so publishing from it re-originated the
  whole swarm's demand table under our own id every 30 s, refreshing timestamps
  that should have been ageing out. `local_region_demand` is the measured half
  and the only one that may be published; the two have different key types, so
  the wrong one no longer compiles at the publish site. Guard:
  `the_demand_we_gossip_is_the_demand_we_measured`. **GossipSub already carries
  the originator's message to every node — re-originating is not what makes it
  travel.**
- **A newcomer is caught up POINT TO POINT.** Gossip cannot be addressed to one
  peer, so answering "someone new connected" with a topic-wide re-announce made
  one join cost every node a full copy of every manifest: inbound on
  `swarm/models` went **98.8 → 398.3 KB/s** after a single node joined, and
  peers reconnect about 7 times an hour on an 8-peer node, roughly doubling the
  rate of full rounds. `NetworkCommand::SendDirectMessage`
  carries any `SwarmMessage` over request_response and the receiver dispatches
  it exactly as a gossiped one, so this needs no new variant and no feature
  bit. The periodic full round stays as the bound on a catch-up that failed.
  **Our capability goes to a newcomer the moment it is identified**
  (`identify.rs`, from `state.local_capability`), not on the manifest tick: a
  peer with no capability is unbounded on every routing rung, and one was handed
  a whole 14B 0.7 s after connecting (#120). Measured: 0.2-10 ms after
  "Peer connected" between two nodes with it, ~5 s from peers without.
  BitTorrent draws the same line: BEP 3's bitfield goes to the peer that
  connected, and only per-piece `have` deltas go to everyone.
- **What the swarm just HEARD is not repeated** (RFC 6206, Trickle, *k* = 1).
  Every holder re-announced its manifests on its own full round, so a model
  held by *k* nodes went out *k* times per round — **87% of an idle node's
  received gossip** (2026-09-23), none of it news. `state.models.manifest_heard`
  holds every `(model, hash)` the whole swarm was gossiped in the window;
  `broadcast_manifests` stays quiet about that exact hash for
  `MANIFEST_QUIET_WINDOW`. ⚠ **Only a GOSSIPED arrival counts** — a point-to-point
  catch-up reached one node, and counting it lets reconnects silence every
  holder. `AuthenticatedMessage.transport` is required for that reason, and
  `note_manifest_heard` ignores `Direct`. ⚠ Record only AFTER verification, and
  never suppress a DIFFERENT hash — a disagreement is information. ⚠ **Keyed
  by VERSION**: remembering only the last hash per model let two disagreeing
  versions (#61) un-hear each other, and ~90% of manifest gossip was holders
  of 9 disputed models re-announcing every round (2026-09-25). Each version
  now goes out once per window, swarm-wide.

## A counter named for an outcome may only be counting the attempt

GossipSub's `sent` / `sent_bytes` count **attempts, once per recipient**:
`msg_sent` runs at the top of `send_message`, before the connected-peer lookup
and before the queue that can refuse the message. So a per-topic figure may
legitimately **exceed** `BandwidthMeter`'s total, and
`sum(topic.sent_bytes) <= out_bytes` must NOT be asserted — the GAP IS THE
SIGNAL, and it is gossip this node was asked to relay and could not.

Reported from the field 2026-09-21 (`swarm/models sent = 389 MB` on a node whose
`out_bytes` was 179 MB). `GossipMeter`'s own doc had claimed these were "the
figure that matches what leaves the interface" — **the comment reasoned about
one quantity while the counter measured another**, which is the timeouts rule's
trap in a new place.

**Both drop paths must be read; they come from different places and are
independently absent.** Queue EXPIRY raises `HandlerEvent::MessageDropped` and
increments `*_messages_dropped_per_topic`; queue FULL touches **no metric family
at all** and leaves the crate only as `gossipsub::Event::SlowPeer`. Reading only
the metrics misses the half that moves on a congested node.

⚠ **`sent_msgs` is not a publish rate** — it includes forwards times recipients.
`published_msgs` is the only field that says whether this node's own timer fires.

⚠ **Second firing of gotcha #673.** GossipSub keeps ~28 metric families; this
node parsed six, and the three that answered the question were among the
twenty-two it did not. **Before estimating the components of a total, look for
the counter you are not reading.**

## A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT

The per-shard tensor table is **~92% of a manifest's bytes**, and a manifest is
86% of inbound gossip — so it is roughly **80% of all gossip traffic**. It is
also a pure function of the GGUF header and ONE integer, so a holder of the
header can rebuild it instead of being sent it. `daemon::shard_loader::derive_tensor_entries`
does; the loader uses it only as a FALLBACK, so a manifest that carries a table
is still used exactly as before.

⚠ **`manifest.shard_count` is the layout's OUTPUT (`layouts.len()`), not its
input.** The publisher passes `div_ceil(total_size, shard_size)`, and
`compute_layer_shard_layouts` can return FEWER shards than asked for — on this
node's 15 models the two differ for 5, always by one. **Feeding the recorded
count back in reproduces 10 of 15 and silently mis-places the other 5.** So the
count is SEARCHED for, and the published per-shard `size_bytes` are the oracle
that says which candidate is the publisher's.

⚠ **Do NOT "fix" the underproduction.** Its own comment claims it cannot
happen, and it does — but every shard FILE in the swarm was cut by the current
behaviour, so changing it would re-split models and make new nodes disagree
with every existing manifest. The behaviour is load-bearing.

⚠ **A wrong offset does not fail — it reads the wrong weights and answers
confidently.** Hence the oracle, and hence `derive_tensor_entries` returns
`None` rather than a partial or unverified table. Verified against all 15 real
models by `a_derived_tensor_table_matches_the_published_one` (ignored; needs
`SWARMLLM_TEST_MODEL_DIR`).

⚠ **A topic is not a message type.** `swarm/models` carries SIX variants, so
its per-topic counters cannot rank two fixes that target different variants on
it. `state.metrics.gossip_by_kind` counts received gossip by
`SwarmMessage::kind_name` — measured 2026-09-21: **`ModelManifest` is 86.4% of
inbound gossip bytes at 32 KB/msg**, `NodeCapabilityUpdate` 3.4%. **So the
manifest is the cost worth attacking and the capability change-gate is not.**
⚠ **Its REPETITION was attacked first (Trickle, above), not its size.** Dropping
the table from the wire is a gossip FORMAT change: ingestion verifies the hash
too, older nodes two hops away would receive it, and the documented route is a
new versioned topic (`docs/invariants/network.md` § "The repetition, not the
size"). ⚠ And a manifest averages **32 KB, not the 13 KB** long carried in the
queue — that came from a per-topic average across all six variants. **An
average over a mixed population is not a figure about any member of it.**
Counted on RECEIVE, because an idle node's upload is relaying and what it
relays is what it received.

→ `docs/invariants/network.md` § "Gossip volume"

⚠ **`NodeCapabilityUpdate` is still broadcast every tick and is NOT
change-gated** — `ram_available_mb`, `disk_available_mb` and `uptime_seconds`
move every tick, so it cannot be gated as it stands without separating the
stable fields from the volatile ones. → `docs/FUTURE_WORK.md` #91.

→ `docs/invariants/network.md`

## Single-source-of-truth helpers — Network protocol, peers and the model registry

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/network.md`

- **`inference::pipeline::remote_generate::StreamReassembler`** — the single place a remote reply's token stream is put back in order.
- **A hole in a peer-served reply is FILLED, not waited out** — `RetainedReplies` keeps each fast-path reply this node streams, and `SwarmMessage::ResendTokens` is answered from it — only to the peer the reply was for.
- **`NodeCapability.cpu`** — a processor described the way a graphics card always has been.
- **`PeerInfo::ack_srtt_ms` is what routing prices a peer by** — written on every acknowledged tensor forward from `AckRttEstimator::srtt_ms`, capped at `ACK_SRTT_ROUTING_CAP_MS` (10 s) because the estimator DOUBLES on a miss.
- **`mem_bandwidth::remeasure_keeping_the_best`** — the memory-bandwidth figure a processor-only node advertises may RISE over its run and never fall.
- **A peer's advertised version may bring the update check FORWARD and may do nothing else** — `update::PeerVersionWatch` on `state.events.peer_versions` wakes `UpdateChecker` through `state.events.update_nudge` (a `Notify`, not a third broadcast channel); it never decides the outcome.
- **`update::SelfUpdateBlocker` — "this node cannot update itself" carries WHY** — `UpdateChecker::self_update_blocker` returns the reason; `key()` is the stable string the dashboard translates across 21 locales, `advice()` the English one the daemon log and `swarmllm update` print.
- **`ModelRegistry::manifests_to_gossip`** — the single answer to "which manifests should this node re-broadcast?": ones it published **and ones it holds a shard of**.
- **`model::manifest::merge_known_shard_hashes`** — the rule that a shard hash may go from unknown to known but never back.
- **`model::manifest::keep_known_hashes_over_contradicting_ones`** — the sibling rule: a hash we already hold is not replaced by a stranger's CONTRADICTING one when the two manifests describe the same shape. Shards this node HOLDS are exempt, because there the claim is testable. Without it the registry oscillated for ever between two peers.
- **`types::slugify_model_name`** — the single derivation of a model id from a human display name.
- **`model::huggingface::is_trusted_publisher`** — canonical curator-allowlist check for an HF `repo_id`.
- **`SharedState::resolve_connected_peer_id_bytes`** — the resolver to use for any message that `network::manager::relay::is_relay_eligible` refuses, i.e. everything except `RemoteGenerateRequest` / `StreamingToken` / `CancelInference`. For those, "reachable" means "connected".
- **`ModelRegistry::describes_a_different_build`** — is this manifest the same FILE as ours, or another build wearing the same name? A model id comes from a display name (`slugify_model_name`), so every independent GGUF build collapses into one identity. Compares SHAPE, never hashes, and is gated on `has_origin_knowledge`.
- **`model::manifest::is_backup_artifact_id`** — canonical check for a model id that is a copied-folder backup (`<model>.FULLBACKUP`, `<model>.old`, `<model>~`, `… copy`) rather than a real model identity. Netted at `ModelRegistry::register_manifest`, the one point every adoption path funnels through.

## ACK-Timeout Fast-Fail for rr Sends

`SendDirectMessage` carries `delivery_request_id: Option<uuid::Uuid>`.
When `Some(uuid)`, the `pending_rr_observability` entry inserted in
`handle_send_rr_message` carries that uuid; the 10s
`RR_ACK_TIMEOUT_SECS` sweep in the network manager closes
`streaming_token_txs[uuid]` if no Response or OutboundFailure event
fires (libp2p rr can silently drop sends under load — observed and
documented). Caller sees Err in ~10–20s instead of `FIRST_TOKEN_TIMEOUT`
(120s). Fire-and-forget rr paths use `None`; streaming paths
(`remote-generate` fast path) MUST set `Some(uuid)`. Pair with the
`is_transient_remote_failure` retry in `dispatch_single` so a single
silent-drop transparently re-routes to a different holder.

## A connection the swarm DENIED is forgotten by request-response (2026-10-02)

The vendored request-response records a connection when it is handed one
(`handle_established_*_connection`), BEFORE the swarm decides to keep it; the
per-peer cap (3) then denies some, reported as `ListenFailure` / `DialFailure`,
never `ConnectionClosed`. **`forget_denied_connection` drops the entry and fails
its requests on both.** Remembered, a ghost had nothing pending and so won every
send made while the real connections each had one outstanding — every burst lost
its tail, silently, until the 600 s timeout. This is the "silent drop under load"
the section above works around. A behaviour that keeps per-connection state must
clean up on those two events as well as on a close.

→ `docs/invariants/network.md` § "A connection the swarm denied is forgotten"

## A split token crosses on the pipeline stream, keyed by (request, peer) (2026-09-27)

`inference.persistent_pipeline_stream` is 2.4x per token on a 412 ms link (1.19 →
2.85 tok/s, A-B-A in one binary) but stays OFF: a healthy peer stalled reading a
prompt pass on it in the 4-node rig, and the path has no receipt ACK (#133).
Streams are keyed by **(request, peer)** — keyed by request alone, a second
remote peer's forward went down the first's stream. Measure a per-message cost
on a REAL link, never loopback — and run the failover rigs before any default flip.

→ `docs/invariants/network.md` § "A split token crosses on the pipeline stream"

## A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted (2026-09-27)

The swarm negotiates every substream it opens with multistream-select **V1Lazy**
(0-RTT for one offered protocol; `SWARMLLM_SUBSTREAM_V1=1` restores V1 for an
A/B). Under V1 each request-response message waited a round trip for the
protocol confirmation, so every split token paid one extra. Wire-identical for
the listener: older peers need nothing.

It halves every exchange sample, the PEX ping included. **`PeerInfo::latency_ms`
and ACK samples are COSTS** — routing, the hand-off bound and deadlines read them
raw and should move. **A reader that means DISTANCE goes through
`network::manager::physical_rtt_ms`**: the LAN heuristic (`exchange_says_lan`,
a privacy boundary — private mode admits LAN peers) and the coordinate
(`observe_network_coord`). Never compare a raw sample to a distance constant.

→ `docs/invariants/network.md` § "A substream sends with its protocol proposal"
