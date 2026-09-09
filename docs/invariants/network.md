# Network protocol, peers and the model registry

The evidence behind the rules in `.claude/rules/architecture.md`: what each
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

## **A peer's advertised version may bring the update check FORWARD and may do

**A peer's advertised version may bring the update check FORWARD and may do
nothing else** (2026-09-03 evening). `update::PeerVersionWatch` on
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
manifest carrying placeholders for exactly those five, one corrupt (gotcha
#381).
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

the resolver to use for
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
