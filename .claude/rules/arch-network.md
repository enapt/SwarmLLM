---
paths:
  - "src/network/**"
  - "src/daemon/dispatch/**"
  - "src/crypto/**"
  - "src/pool/**"
  - "src/update.rs"
  - "src/model/manifest.rs"
  - "src/model/registry.rs"
  - "src/model/acquisition.rs"
  - "src/model/distribution.rs"
  - "crates/swarmllm-types/**"
  - "vendor/libp2p-request-response/src/**"
  - "src/model/huggingface/**"
  - "src/model/shard.rs"
  - "src/model/lora.rs"
  - "src/identity/**"
  - "src/model/mod.rs"
---

# Network protocol, peers, shards and the model registry

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## A disconnect retires a session key; it must not destroy it — and must not keep it

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

## A downloaded shard is hashed OFF the event loop

`verify_shard` on a whole shard is ~200 ms of BLAKE3 for 500 MB; on the swarm
event loop it stalls every ping, gossip message and tensor forward (#108). The
network manager hashes on the blocking pool and posts the verdict back
(`shard_verdict_tx` → `requests.rs::finish_p2p_shard`, ON the loop). Guard:
`the_network_loop_never_hashes_a_shard_inline`.

→ `docs/invariants/network.md` § "A downloaded shard is hashed off the event loop"

## One writer per shard file, and finishing one shard says nothing about the others

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

## Destroying a shard we hold needs better evidence than a stranger's claim

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

## A holder claim means VERIFIED, not transferred

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

## The activity list reports a TRANSITION; the log may report every message

`emit_activity`'s ring is 100 entries, and it is also the replay a dashboard
opens on and the "recent activity" of the pasteable report. An event emitted per
protocol message empties it of everything the user did — measured at 102 of 114
entries. Before adding an ActivityEvent, ask what makes it fire: if that is a
peer's timer or a handler libp2p re-runs, gate it on a state CHANGE and prefer
one entry per event over one per item inside it. Three sites had this wrong at
once, each with the identical lesson already written down for the log line
beside it. The per-message DIAG log is the durable record and stays.

→ `docs/invariants/network.md`

## A holder record names a BUILD, not just a shard

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

## A result names the step it answers; one that did not arrive is sent again (2026-09-25)

A request's forwards to one segment share its id, so a result matched by id alone
cannot tell step N from N+1 — a late or resent copy would answer the NEXT step,
silently. `LayerResult::answers_step` (the `0x07` trailer, always LAST) names the
forward — position AND layer range, because one node can serve two segments at
one position; `PendingLayerResult::expects_step` (`ExpectedStep`) refuses any
other. **Every registration sets it from the forward it actually sends** (every
hop's range for a chain; a failover replay waits on 0). `sequence_num` is a
prompt-pass flag, not a counter. Only then may a serving node
resend a result on `OutboundFailure` (`resend_lost_result`, gated on the
coordinator's `features::RESULT_STEP`, once). Test with `SWARMLLM_FAULT_RESULT=lose|duplicate`.

→ `docs/invariants/network.md` § "A result names the step it answers"

## Every reply a serving node sends goes to whoever is WAITING — failures included

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

## Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind (2026-09-26)

A forward was acknowledged on arrival, so dropping it at the dispatcher's caps
cost the coordinator its whole segment deadline. **Every kind of peer work takes
a `PeerWorkSlot`** (forwards, whole-model generations, image encodes — generations
were uncounted and could hold every permit), and a refusal is **answered**:
`layer_forward::refuse_forward` / `remote_generate::refuse_request`, worded by
`peer_work_refusal()` so the coordinator bars the peer and re-plans without
retracting its shards. A spawned refusal carries the ADDRESS, never the payload.
An image-encode refusal still cannot be said (no error field; #123).

→ `docs/invariants/network.md` § "Work a serving node will not run is refused OUT LOUD"

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

## Whether a peer is on our LAN is decided only from what WE observed

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

## Completing libp2p Identify does not make a peer one of ours

**`network::manager::identify::peer_speaks_swarmllm`** is the single answer to
"is this libp2p node a SwarmLLM node?", and it is gated at the one point where
Identify turns a connection into a peer — BEFORE the Kademlia insert, so a
foreign node never enters the routing table either.

→ `docs/invariants/network.md`

## One dial per PEER, never one per address

**`NetworkManager::dial_bootstrap_peers` + `discovery::plan_bootstrap_dials`**
are how bootstrap and cached addresses are dialled. Group by target peer, one
`DialOpts::peer_id(..).addresses(all)` each, through `dial_checked`, with
`PeerCondition::DisconnectedAndNotDialing`.

→ `docs/invariants/network.md`

## Peer Cache: storable vs dialable

`network/peer_cache.rs` answers two different questions and they must not be
conflated:

→ `docs/invariants/network.md`

## ModelRegistry Holder Counts

**`merge_dht_providers` is the one writer of `shard_holders` that cannot remove
a holder** — it loops `record_shard_holder` over a DHT `GetProviders` result.
That matters because a provider record outlives the fact it asserts: libp2p-kad
keeps one for 24 h, republishes at 12 h, and other peers serve it, so a node that
deleted or lost a shard is still advertised as holding it for hours. An
add-only writer wins every disagreement with a writer that removes, and its
cadence decides how fast.

→ `docs/invariants/network.md`

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

→ `docs/invariants/network.md`

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

## Network coordinates: publish them rough, feed them a MINIMUM

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
