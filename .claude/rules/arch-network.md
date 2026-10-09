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

`SessionManager::remove_session` retires the live key (openable, never sealable, for `PREVIOUS_KEY_GRACE`) on EVERY full disconnect, with no exemption — not even mid-pipeline (`a_disconnect_retires_the_session_even_mid_pipeline`). A failed `open` calls `request_rekey` (inside `open`, so every decrypt site inherits it), including when we hold no session at all; the refused forward is resent once the repair lands (`ForwardRefusal::Undecryptable`, `0x06`, gated on `features::FORWARD_REFUSAL_REASON`; `arch-scheduling.md`).

→ `docs/invariants/network.md` § "A disconnect retires a session key; it must not destroy it"

## A key derived while ANSWERING an exchange waits until the peer proves it has it

A responder never installs a key it derived while answering an exchange until the peer proves it has it: `accept_ephemeral_exchange` parks it in `unconfirmed`, `open` promotes it. The initiator's `SESSION_CONFIRM_MARKER` comes from `complete_ephemeral_session` so seal and install cannot come apart. Gated on `features::SESSION_KEY_CONFIRM`; WireGuard's rule.

→ `docs/invariants/network.md` § "A key derived while ANSWERING an exchange waits until the peer proves it has it"

## A downloaded shard is hashed OFF the event loop

`verify_shard` on a whole shard is ~200 ms of BLAKE3 for 500 MB; on the swarm
event loop it stalls every ping, gossip message and tensor forward (#108). The
network manager hashes on the blocking pool and posts the verdict back
(`shard_verdict_tx` → `requests.rs::finish_p2p_shard`, ON the loop). Guard:
`the_network_loop_never_hashes_a_shard_inline`.

→ `docs/invariants/network.md` § "A downloaded shard is hashed off the event loop"

## One writer per shard file, and finishing one shard says nothing about the others

One writer per `shard_NNN.bin.tmp`: **`ModelMgmt::claim_shard_download`** is the RAII exclusion; **`SharedState::remove_acquisition_if_idle`** is the only tidy-up removal of a progress entry (`a_finished_download_does_not_delete_the_progress_of_one_still_running`); **`ModelMgmt::live_cancel_flag`** is the one source of a cancel flag. Only a download itself deletes its partial file (`cleanup_tmp_files_no_one_is_writing` skips claimed shards).

→ `docs/invariants/network.md` § "One writer per shard file, and finishing one shard says nothing about the others"

## Destroying a shard we hold needs better evidence than a stranger's claim

**`ModelRegistry::mismatch_policy`** is the single answer to "may this node quarantine its own copy?" — only when `expected` is the hash from the model's ORIGIN. `ShardStore::verify_shard` takes `OnMismatch` as a required parameter. A gossiped hash is a claim about the claimant's build; a disagreement is kept and served, NOT settled (`docs/FUTURE_WORK.md` § "A disputed shard is kept but the disagreement is never settled"). The one deletion on peers' word is the heal on a node with no origin to ask (`settle_by_checked_holders`, below), and its bar is higher: two connected holders that CHECKED against the origin agree, none holds our bytes.

→ `docs/invariants/network.md` § "Destroying a shard we hold needs better evidence than a stranger's claim"

## A holder claim means VERIFIED, not transferred

Nothing records a peer — or this node — as holding a shard until its BLAKE3 check has passed. **`daemon::dispatch::progress_claims_the_peer_holds_it`** is the single reading of a `ShardDownloadProgress` and requires `DownloadState::Complete`; a percentage never asserts holding (`acquisition::maybe_broadcast_shard_progress` caps at `IN_FLIGHT_MAX_PCT`). Gotchas #78, #184.

→ `docs/invariants/network.md` § "A holder claim means VERIFIED, not transferred"

## The activity list reports a TRANSITION; the log may report every message

The 100-entry `emit_activity` ring is the dashboard replay and the report's recent activity: gate an ActivityEvent on a state CHANGE, one entry per event not per item; the per-message DIAG log stays.

→ `docs/invariants/network.md` § "The activity list reports a TRANSITION; the log may report every message"

## A holder record names a BUILD, not just a shard

**`ModelRegistry::shard_holders`** is the single read accessor and filters out holders of a different build (`expected_build_tag`); `all_shard_entries` is the RAW map — never show or decide on a holder count from it. ⚠ "Holds a part" (`peers_hosting`) and "could serve it alone" (`peers_complete`) are two counts. `ModelPeerCounts` carries `servable` and `other_build` together. Guard: `a_holder_count_shown_to_a_person_is_the_count_that_can_serve`.

→ `docs/invariants/network.md` § "A holder record names a BUILD, not just a shard"

## One upload per model id, on every node (2026-10-02)

**`model::canonical`** is the single answer to "which file IS this model". `hf_sources` is written only through `note_origin_claim` / `adopt_canonical_build`; a header only through `fetch_model_header`; every download path asks `canonical_allows_acquisition`; `auto_manage::canonical` heals a node holding another upload. ⚠ **Never reorder `TRUSTED_HF_PUBLISHERS`** — its order is a swarm-wide contract (#151). The heal DELETES parts that are not the upload's bytes (another layout, a failed 64 KB check, a dispute) and re-fetches the upload's through the repair queue — peers when the part's hash is known, HuggingFace otherwise (`replace_parts`); nothing is deleted unless HuggingFace answered that pass, nor while the model is in use. A peer's manifest goes through `SharedState::judge_peer_manifest` (another upload, or another build while a download here is under way — #158); the newcomer catch-up carries the claim with the manifest (`health::monitor::upload_claim`).

**A node vouches only for bytes that are the swarm's upload (2026-10-03):** an announced tag is the tag of the BYTES (`ModelRegistry::announced_build_tag`; a dispute is written only by `SharedState::note_shard_disputed`, with the check's verdict); a copy waiting to be replaced is WITHHELD — `SharedState::note_canonical_holding` is the one writer of `canonical_holding` and of `ModelRegistry::set_model_withheld`, honoured by `shard_announce`, `manifests_to_gossip`, the capability and shard serving; a part entering from a peer is byte-checked against the upload (`auto_manage::canonical::part_is_from`), and the heal re-checks every part that changed (`CheckedParts`).

**A node with no origin to ask is healed by the holders that checked theirs (#160):** `ShardAnnounce::origin_checked_models` says which copies the sender's heal compared with the upload THIS run — only `Holding::Canonical` (`note_canonical_holding` → `set_model_origin_checked`), never a copy settled by peers; `HolderRecord::checked` changes only on the holder's own announcement. Offline mode or no HuggingFace → `settle_by_checked_holders`: a part goes only when `CHECKED_QUORUM` (2) CONNECTED checked holders agree and none holds ours, fetched against the full hash their tag names (`heard_part_hashes`, recorded before any merge). Where HuggingFace answers, checked holders' disagreement is a dispute the UPLOAD settles. A `source_path` exempts parts only while its file exists.

→ `docs/invariants/network.md` § "One upload per model id, on every node"

## ACK-Timeout Fast-Fail for rr Sends

Streaming rr sends MUST set `SendDirectMessage.delivery_request_id = Some(uuid)`; the `RR_ACK_TIMEOUT_SECS` sweep then closes `streaming_token_txs[uuid]` when libp2p drops a send silently. Pair with the `is_transient_remote_failure` retry in `dispatch_single`.

→ `docs/invariants/network.md` § "ACK-Timeout Fast-Fail for rr Sends"

## A connection the swarm DENIED is forgotten by request-response (2026-10-02)

**`forget_denied_connection`** drops a request-response connection the swarm denied (per-peer cap) and fails its requests. Anything keeping per-connection state must clean up on `ListenFailure` / `DialFailure` as well as on a close.

→ `docs/invariants/network.md` § "A connection the swarm DENIED is forgotten by request-response"

## A tensor forward is acknowledged on receipt; a result is always its own request (2026-08-21)

`requests.rs` answers an inbound `LayerForward` with `SwarmResponse::Ack` on decode, BEFORE any work, from the network manager; a result is always its own request (`tensors::handle_send_tensor_result` → `send_tensor_result_as_request`). The fast-fail (`forward_ack_deadline_secs`) is gated ONLY on the peer's `features::FORWARD_ACK` bit and never reaps a slow answer — the compute deadline stays with the pipeline (gotcha #354).

→ `docs/invariants/network.md` § "A tensor forward is acknowledged on receipt; a result is always its own request"

## A split token crosses on the pipeline stream, keyed by (request, peer) (2026-09-27)

`inference.persistent_pipeline_stream` is measured faster but stays OFF (no receipt ACK, #133). Streams are keyed by **(request, peer)**. Measure per-message cost on a REAL link; run the failover rigs before any default flip.

→ `docs/invariants/network.md` § "A split token crosses on the pipeline stream, keyed by (request, peer)"

## A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted (2026-09-27)

Substreams negotiate with multistream-select V1Lazy (`SWARMLLM_SUBSTREAM_V1=1` restores V1 for an A/B). `PeerInfo::latency_ms` and ACK samples are COSTS; a reader that means DISTANCE goes through **`network::manager::physical_rtt_ms`** (`exchange_says_lan`, `observe_network_coord`) — never compare a raw sample to a distance constant.

→ `docs/invariants/network.md` § "A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted"

## A speculative verify is walked where the logits are (2026-09-27)

A verify's last segment walks the drafts and answers with token ids (`LayerForward::spec_walk_at_tail`, gated at the SENDER on `features::SPEC_WALK_AT_TAIL`). **`sampling::sampled_accept_reject`** is the one rule; **`pipeline::VerifyReply::accept`** the one place a reply becomes accepted tokens; **`layer_forward::spec_trailer_flags`** the one writer of the flags byte. Shared noise (`coupling_seed`, `0x0B`, `features::COUPLED_SAMPLING`) is keyed by ABSOLUTE position, never relative (`a_drafter_keyed_at_the_samplers_positions_is_accepted_and_one_off_is_not`).

→ `docs/invariants/network.md` § "A speculative verify is walked where the logits are"

## A check travels a chain like a decode step does (2026-10-02)

**`PipelineExecutor::verify_may_chain`** decides whether `forward_verify_through_segments` sends a run of remote segments ONE forward (every hop advertising `features::CHAINED_VERIFY`, gated at the coordinator); a chained check resends on nothing (`ResendOnRefusal::Never`). `SWARMLLM_CHAIN_VERIFY=0` is the control arm.

→ `docs/invariants/network.md` § "A check travels a chain like a decode step does"

## A result names the step it answers; one that did not arrive is sent again (2026-09-25)

`LayerResult::answers_step` (`0x07`) names the forward; `PendingLayerResult::expects_step` refuses any other, and every registration sets it from the forward it actually sends. Only then may a serving node resend a lost result (`resend_lost_result`, gated on `features::RESULT_STEP`). Test with `SWARMLLM_FAULT_RESULT=lose|duplicate`.

→ `docs/invariants/network.md` § "A result names the step it answers"

## A stream of verifies runs in its order, and its answers are found by number (2026-09-30)

Streamed verify forwards (`LayerForward::stream_seq`, `0x0C`, gated on `features::STREAMED_VERIFY`) run in number order via **`daemon::state::forward_streams`**; `pending_layer_results` is keyed by `WaiterKey` and an answer echoes its number (`0x08`). A stream is ONE piece of its sender's work (`dispatch::StreamWorkSlot`, `MAX_STREAM_CHUNKS_HERE`); every admission refusal goes through `refuse_forward`. Stream only to a peer advertising `features::STREAM_AS_ONE_WORK`. **`dsd_stream::stream_shape`** takes any plan with ONE peer segment (the boomerang too, #152): where this node holds the last layers the walk is ours and those segments run as each answer is TAKEN, never as it arrives (#180).

→ `docs/invariants/network.md` § "A stream of verifies runs in its order, and each answer names its number"

## Every reply a serving node sends goes to whoever is WAITING — failures included

**`layer_forward::reply_target`** is the single answer to who is WAITING; it returns a `ReplyTo` nothing else can build, so `send_error_result` cannot be handed the raw sender (the previous hop in a chain). Gotcha #707.

→ `docs/invariants/network.md` § "A chained hop's refusal goes to the coordinator"

## Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind (2026-09-26)

Every kind of peer work takes a `PeerWorkSlot`, and a refusal is ANSWERED — `layer_forward::refuse_forward` / `remote_generate::refuse_request`, worded by `peer_work_refusal()` — never silently dropped. A request that has not started leaves the last quarter of the slots alone (`admits_a_new_request`). Image-encode refusals still cannot be said (FUTURE_WORK #123).

→ `docs/invariants/network.md` § "Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind"

## UPnP that the router refuses is said out loud, and a stopping node hands its ports back (2026-10-08)

UPnP is the DIRECT `libp2p-upnp` ≥ 0.6 dependency (it backs off a refused mapping; the facade's 0.5.0 asked again without pause, for ever) — the facade's `upnp` feature stays OFF; guard `upnp_is_the_release_that_backs_off`. A refused mapping emits no event, so **`network::manager::upnp_watch::UpnpWatch`** explains the silence once (`UPNP_QUIET_AFTER`), and `release_upnp_mappings` closes the P2P listeners at a clean shutdown so the router gets the ports back (#239).

→ `docs/invariants/network.md` § "UPnP that the router refuses is said out loud"

## "Is this connection direct?" — `network::relay::addr_is_direct_transport`

**`network::relay::addr_is_direct_transport`** is the single answer, for both layers that choose a connection: a real `/ip4`, `/ip6` or `/dns*` hop AND no circuit. `is_relay_circuit_addr` alone misses the listener's view; the vendored request-response inbound handler records the send-back address. `peer_direct_conns` is gated on it; any new "prefer direct" logic uses it (gotchas #356, #179). Keep the loop-stall tripwire in `NetworkManager::run`.

→ `docs/invariants/network.md` § ""Is this connection direct?" — `network::relay::addr_is_direct_transport`"

## Whether a peer is on our LAN is decided only from what WE observed

`is_lan_peer` is a privacy boundary: never derive it from `identify::Info` (peer-controlled); use only the connection's own address, mDNS, and a measured RTT. ⚠ The flag is sticky. Guard: `lan_membership_is_never_decided_from_what_a_peer_told_us`.

→ `docs/invariants/network.md` § "LAN membership is decided only from what we observed"

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

`network/peer_cache.rs` answers two different questions; never use one as the
other. **`filter_storable`** — what is worth KEEPING (`save_peer_cache`): keeps a
peer's private addresses wherever this node is now. **`filter_dialable`** — what
is worth dialling FROM HERE (every dial path): also drops a peer's private
addresses when we have none, or when the peer advertises a public one (the
Docker `172.17.0.1` loop-back).

→ `docs/invariants/network.md` § "Peer Cache: storable vs dialable"

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

**`network::redact::redact_addresses`** hides every host in the diagnostics report at the one point `api::admin::diagnostics` returns; `?full=1` is the opt-in. Guard: `the_diagnostics_report_hides_addresses_unless_full_is_asked_for`.

→ `docs/invariants/network.md` § "A report built to be handed to a stranger is a publishing surface"

## Centralised Wire-Format Helpers

Single sources of truth for invariants that silently break at the wire if duplicated. Read the evidence before changing one.

- **`network::protocol::build_layer_forward_aad`** — ⚠ a new trailer is NOT a no-op for an older peer: feature-gate every optional trailer at the SENDER, and bind it in this helper (encrypt and `decode_layer_forward_encrypted` both go through it); clamp a clamped trailer AFTER the AAD is rebuilt.
- **`network::pipeline_stream::chunk_layer_forward`** — the one splitter for chunked sends.
- **`SharedState::resolve_pending_layer_result`** — the ONLY way to deliver a `LayerResult` into `pending_layer_results`.
- **`daemon::dispatch::timestamp_fresh_one_sided`** — the staleness primitive; `gossip_timestamp_fresh` wraps it.
- **`credit::ledger::check_signed_freshness`** — staleness for signed `DateTime<Utc>` messages.
- **`pipeline::pack_verify_tokens_to_le_bytes`**, **`pipeline::build_spec_verify_forward`**, **`pipeline::build_kv_truncate_forward`**, **`pipeline::register_pending_layer_result`** — the shared builders for verify/truncate forwards and waiter registration.
- **`storage::Database::with_write_table`** — the write-transaction wrapper.
- **`swarmllm_types::ShardResponse::empty()`**, **`swarmllm_types::LayerResult::error(request_id, reason)`** — canonical refusal/error replies.
- **`network/manager/connections::try_enqueue_redial`** — dedup + cap + push for `pending_redial`.
- **`responses::types::raw_tool_kind_or_unknown`**, **`cli::bail_if_no_api_key` / `cli::exit_daemon_unreachable`** — shared error wording.
- **`model::auto_manage::spawn_check_and_load`** — "shard landed → reload model → refresh dashboard", always the three steps together.
- **`pool::invite::{encode_invite_code, decode_invite_code}`** — the `swarmpool://` v2 codec; `looks_like_v2` routes v2 from the legacy path.

→ `docs/invariants/network.md` § "Centralised Wire-Format Helpers"

## Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer

Publish on CHANGE, never per item per tick — gate on a digest of what is ASSERTED, excluding any timestamp. Gossip only what this node MEASURED (`local_region_demand`; guard `the_demand_we_gossip_is_the_demand_we_measured`). Catch a newcomer up POINT TO POINT (`NetworkCommand::SendDirectMessage`), and send our capability at identify time. Don't repeat what the swarm just HEARD (`state.models.manifest_heard`, `MANIFEST_QUIET_WINDOW`; only GOSSIPED arrivals count, keyed by VERSION).

→ `docs/invariants/network.md` § "Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer"

## A counter named for an outcome may only be counting the attempt

GossipSub's `sent` / `sent_bytes` count ATTEMPTS per recipient, so `sum(topic.sent_bytes) <= out_bytes` must NOT be asserted — the gap is the signal. Read both drop paths (queue expiry via metrics; queue full only via `gossipsub::Event::SlowPeer`). `sent_msgs` is not a publish rate; `published_msgs` is. Before estimating a total's components, look for the counter you are not reading (gotcha #673).

→ `docs/invariants/network.md` § "A counter named for an outcome may only be counting the attempt"

## A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT

The per-shard tensor table is ~92% of a manifest and is derived, not sent: **`daemon::shard_loader::derive_tensor_entries`** rebuilds it as a FALLBACK. ⚠ `manifest.shard_count` is the layout's OUTPUT, so the count is SEARCHED for with the published `size_bytes` as oracle; return `None` rather than a partial table; do NOT "fix" the underproduction. `a_derived_tensor_table_matches_the_published_one` (ignored; `SWARMLLM_TEST_MODEL_DIR`). `NodeCapabilityUpdate` is not change-gated (FUTURE_WORK #91).

→ `docs/invariants/network.md` § "A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT"

## Network coordinates: publish them rough, feed them a MINIMUM

`NodeCapability.coord` (Vivaldi, `features::NETWORK_COORDS`) is published however rough (`a_brand_new_node_still_publishes_its_coordinate`; `is_usable()` is the CONSUMER's gate). Feed the windowed MINIMUM, never a raw sample — **`SharedState::observe_network_coord`** is the single writer, `LatencyFilter` takes the minimum (not Serf's median). Size a time window against its fill rate: `LATENCY_WINDOW_MIN_SAMPLES`, `LATENCY_SAMPLE_MAX_AGE_MS`.

→ `docs/invariants/network.md` § "A network coordinate must be published rough, and fed a MINIMUM"

## Single-source-of-truth helpers — Network protocol, peers and the model registry

Each names the ONE place a decision is made; a second implementation is this codebase's most-repeated defect (`.claude/rules/architecture.md` § "One invariant, N paths"). Read the topic file before changing one.

- **`SharedState::publicly_reachable`** — is this node reachable from the internet (a confirmed public external address). `node_stats.nat_status` is the last per-address NAT event, never the answer (`api::admin::nat_headline`, gotcha #796).
- **`inference::pipeline::remote_generate::StreamReassembler`** — puts a remote reply's token stream back in order.
- **A hole in a peer-served reply is FILLED, not waited out** — `RetainedReplies`, `SwarmMessage::ResendTokens`.
- **`NodeCapability.cpu`** — a processor described like a graphics card.
- **`PeerInfo::ack_srtt_ms` is what routing prices a peer by** — capped at `ACK_SRTT_ROUTING_CAP_MS`.
- **`mem_bandwidth::remeasure_keeping_the_best`** — may rise, never fall.
- **A peer's advertised version may bring the update check FORWARD and may do nothing else** — `update::PeerVersionWatch`.
- **`update::SelfUpdateBlocker` — "this node cannot update itself" carries WHY** — `key()` / `advice()`.
- **`ModelRegistry::manifests_to_gossip`** — which manifests to re-broadcast.
- **`model::manifest::merge_known_shard_hashes`** — a hash goes unknown → known, never back.
- **`model::manifest::keep_known_hashes_over_contradicting_ones`** — a held hash is not replaced by a contradicting one.
- **`types::slugify_model_name`** — model id from a display name.
- **`model::huggingface::is_trusted_publisher`** — curator-allowlist check.
- **`SharedState::resolve_connected_peer_id_bytes`** — for messages `network::manager::relay::is_relay_eligible` refuses.
- **`ModelRegistry::describes_a_different_build`** — same FILE or another build wearing the name; SHAPE, never hashes.
- **`model::manifest::is_backup_artifact_id`** — copied-folder backup ids, netted at `ModelRegistry::register_manifest`.
- **`ModelManifest::context_length`** — the model's declared context, OUTSIDE `manifest_hash` (like `mmproj`; a hashed field would make old and new nodes disagree on every manifest); filled by `SharedState::fill_declared_contexts` before gossip, kept unknown → known by `register_manifest` (#189).

→ `docs/invariants/network.md` § "Single-source-of-truth helpers — Network protocol, peers and the model registry"
