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
