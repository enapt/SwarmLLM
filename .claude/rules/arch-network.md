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

`SessionManager::remove_session` retires the live key (openable, never sealable, for `PREVIOUS_KEY_GRACE`) on EVERY full disconnect, with no exemption — not even mid-pipeline (`a_disconnect_retires_the_session_even_mid_pipeline`). A failed `open` calls `request_rekey` (inside `open`, so every decrypt site inherits it), including when we hold no session at all; the refused forward is resent once the repair lands (`ForwardRefusal::Undecryptable`, `0x06`, gated on `features::FORWARD_REFUSAL_REASON`; `arch-scheduling.md`). The refusal names its forward (`LayerForward::step`, as every answer does): a streamed forward's wait is keyed by its number, and an unnamed refusal reached no one (#303). **An age eviction retires the same way** (`evict_stale` returns the peers, and the rotation task re-keys each at once): a session ages out only when no rotation renewed it, and the peer may still be sealing with it (#304).

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

## The network loop never waits for the disk (2026-10-20, #314)

A database commit waits for `fdatasync`. **`storage::Database::with_write_table`** runs every write through `without_starving_the_runtime`, so a task that commits first hands its worker's queue — the LIFO slot the network loop was just woken into, which no other worker may steal — to another thread. The loop's OWN writes go to the blocking pool (`peer_cache::CacheWriter`, newest snapshot wins). Every arm starts with **`loop_stall::ArmStart::now()`**, so a stall line says what the arm's THREAD did (on/off the processor, disk page faults, waits, preemptions) — the log append is NOT a stall (measured, 2026-10-20). Guards `a_commit_does_not_hold_the_task_its_worker_just_woke`, `the_cache_writer_keeps_the_newest_snapshot`. **Nor does it parse a model header** (2026-10-31): the capacity snapshot it refreshes at every Identify asks `SharedState::refused_model_if_learned`, never `refused_model` / `gguf_meta_for`; the health tick learns missing geometry on its blocking thread (`learn_geometry_of_known_models`) — 84 of 89 stalls were that parse at a start. Test `the_snapshot_reads_no_header_and_the_tick_learns_it`. **Nor waits on a lock held through a commit** (2026-11-02): no guard on a map the loop touches (`peer_registry` above all) lives across a database write — change under the lock, write after it, many rows in ONE commit (`Database::put_json_many`); `TrustManager::decay_all` committed per peer inside `iter_mut`. Test `a_decay_pass_commits_once_with_no_registry_entry_locked`.

→ `docs/invariants/network.md` § "The network loop never waits for the disk"

## One writer per shard file, and finishing one shard says nothing about the others

One writer per `shard_NNN.bin.tmp`: **`ModelMgmt::claim_shard_download`** is the RAII exclusion; **`SharedState::remove_acquisition_if_idle`** is the only tidy-up removal of a progress entry (`a_finished_download_does_not_delete_the_progress_of_one_still_running`); **`ModelMgmt::live_cancel_flag`** is the one source of a cancel flag. Only a download itself deletes its partial file (`cleanup_tmp_files_no_one_is_writing` skips claimed shards) — so EVERY writer claims from its first byte: a peer transfer through `handle_send_shard_request`'s REQUIRED `PartAsk` (`Starts` → **`ModelMgmt::hold_for_a_peer_transfer`**; a second start is refused only while the first waits on an answer AND holds its slot — `earlier_transfer`; any other record is forgotten), and it writes only while its slot is parked (**`p2p_transfer_still_owns_the_file`**, claimed by another or by nobody); a failed verdict releases the slot (#418). A staging name no part owns has no claim to ask: the mid-life sweep leaves it to the startup sweep, and its writer removes it on failure (`StagingFile`, #419). A transfer carries on from its bytes (`PartTransfer`, `resume_from` — never past the partial file's length) and a failed check blames only a SOLE sender (`PartTransfer::sole_sender`, #191).

→ `docs/invariants/network.md` § "One writer per shard file, and finishing one shard says nothing about the others"

## Destroying a shard we hold needs better evidence than a stranger's claim

**`ModelRegistry::mismatch_policy`** is the single answer to "may this node quarantine its own copy?" — only when `expected` is the hash from the model's ORIGIN. `ShardStore::verify_shard` takes `OnMismatch` as a required parameter. A gossiped hash is a claim about the claimant's build; a disagreement is kept and served, NOT settled (`docs/FUTURE_WORK.md` § "A disputed shard is kept but the disagreement is never settled"). The one deletion on peers' word is the heal on a node with no origin to ask (`settle_by_checked_holders`, below), and its bar is higher: two connected holders that CHECKED against the origin agree, none holds our bytes.

→ `docs/invariants/network.md` § "Destroying a shard we hold needs better evidence than a stranger's claim"

## A holder claim means VERIFIED, not transferred

Nothing records a peer — or this node — as holding a shard until its BLAKE3 check has passed. **`daemon::dispatch::progress_claims_the_peer_holds_it`** is the single reading of a `ShardDownloadProgress` and requires `DownloadState::Complete`; a percentage never asserts holding (`acquisition::maybe_broadcast_shard_progress` caps at `IN_FLIGHT_MAX_PCT`). Gotchas #78, #184. **`SharedState::parts_servable_here`** is the one answer to which parts this node can serve from disk (the loader, a whole-model hand-off, the request gate): credited here, no download under way, and the file EXACTLY its manifest size — **`shard::shard_file_is_whole`**, the one size verdict, never "within 10%" (#334 (1)); its base **`SharedState::parts_landed_here`** (no writer CLAIM, the progress map's marks unread) is what the acquisition job reconciles against every 10 s, its own marks being the ones the other would wait out (#420). An explicit download asks it too, BEFORE it registers the download, and never fetches such a part again unless disputed or withheld (`admin_hf::shards::parts_already_here`, #393).

→ `docs/invariants/network.md` § "A holder claim means VERIFIED, not transferred"

## The activity list reports a TRANSITION; the log may report every message

The 100-entry `emit_activity` ring is the dashboard replay and the report's recent activity: gate an ActivityEvent on a state CHANGE, one entry per event not per item; the per-message DIAG log stays.

→ `docs/invariants/network.md` § "The activity list reports a TRANSITION; the log may report every message"

## A holder record names a BUILD, not just a shard

**`ModelRegistry::shard_holders`** is the single read accessor and filters out holders of a different build (`expected_build_tag`); `all_shard_entries` is the RAW map — never show or decide on a holder count from it. ⚠ "Holds a part" (`peers_hosting`) and "could serve it alone" (`peers_complete`) are two counts. `ModelPeerCounts` carries `servable` and `other_build` together. Guard: `a_holder_count_shown_to_a_person_is_the_count_that_can_serve`.

→ `docs/invariants/network.md` § "A holder record names a BUILD, not just a shard"

## One upload per model id, on every node (2026-10-02)

**`model::canonical`** is the single answer to "which file IS this model". `hf_sources` is written only through `note_origin_claim` / `adopt_canonical_build`; a header only through `fetch_model_header`; every download path asks `canonical_allows_acquisition`; `auto_manage::canonical` heals a node holding another upload. ⚠ **Never reorder `TRUSTED_HF_PUBLISHERS`** — its order is a swarm-wide contract (#151). The heal DELETES parts that are not the upload's bytes (another layout, a failed 64 KB check, a dispute) and re-fetches the upload's through the repair queue — peers when the part's hash is known, HuggingFace otherwise (`replace_parts`); nothing is deleted unless HuggingFace answered that pass, nor while the model is in use. A peer's manifest goes through `SharedState::judge_peer_manifest` (another upload, or another build while a download here is under way — #158); the newcomer catch-up carries the claim with the manifest (`health::monitor::upload_claim`).

**A node vouches only for bytes that are the swarm's upload (2026-10-03):** an announced tag is the tag of the BYTES (`ModelRegistry::announced_build_tag`; a check's verdict reaches the dispute only through `SharedState::note_shard_disputed` / `note_shard_verified`, judged against the hash the manifest names NOW — `ModelRegistry::note_bytes_checked`; a pass never clears a dispute outright, the startup sweep's did against a hash gossip had replaced, #224); a copy waiting to be replaced is WITHHELD — `SharedState::note_canonical_holding` is the one writer of `canonical_holding` and of `ModelRegistry::set_model_withheld`, honoured by `shard_announce`, `manifests_to_gossip`, the capability and shard serving; a part entering from a peer is byte-checked against the upload (`auto_manage::canonical::part_is_from`), and the heal re-checks every part that changed (`CheckedParts`).

**A node with no origin to ask is healed by the holders that checked theirs (#160):** `ShardAnnounce::origin_checked_models` says which copies the sender's heal compared with the upload THIS run — only `Holding::Canonical` (`note_canonical_holding` → `set_model_origin_checked`), never a copy settled by peers; `HolderRecord::checked` changes only on the holder's own announcement. Offline mode or no HuggingFace → `settle_by_checked_holders`: a part goes only when `CHECKED_QUORUM` (2) CONNECTED checked holders agree and none holds ours, fetched against the full hash their tag names (`heard_part_hashes`, recorded before any merge). Where HuggingFace answers, checked holders' disagreement is a dispute the UPLOAD settles. A `source_path` exempts parts only while its file exists.

**An origin hash ends with the bytes it vouched for (2026-10-14, #270):** the heal forgets the origin hashes of a model with no part here and none on its way (`forget_origin_hashes_of_copies_gone_from_here`) — kept, they made `register_manifest` refuse the swarm's manifest for a copy long pruned — and judges every directory under `models/`, metadata-only ones and their DISK manifest included (`models_kept_on_disk`). Never make the refusal itself ask about held parts: origin hashes load before any manifest so a held copy is protected while the node boots (#384).

→ `docs/invariants/network.md` § "One upload per model id, on every node"

## ACK-Timeout Fast-Fail for rr Sends

Streaming rr sends MUST set `SendDirectMessage.delivery_request_id = Some(uuid)`; the `RR_ACK_TIMEOUT_SECS` sweep then closes `streaming_token_txs[uuid]` when libp2p drops a send silently. Pair with the `is_transient_remote_failure` retry in `dispatch_single`.

→ `docs/invariants/network.md` § "ACK-Timeout Fast-Fail for rr Sends"

## A connection the swarm DENIED is forgotten by request-response (2026-10-02)

**`forget_denied_connection`** drops a request-response connection the swarm denied (per-peer cap) and fails its requests. Anything keeping per-connection state must clean up on `ListenFailure` / `DialFailure` as well as on a close.

→ `docs/invariants/network.md` § "A connection the swarm DENIED is forgotten by request-response"

## A request failure closes a connection only when it means silence (2026-10-16, #290)

**`connections::failure_counts_as_silence`**, asked inside `note_rr_failure`: only `Timeout` (and `UnsupportedProtocols`) count. A connection the transport lost fails every in-flight request at once, BEFORE `ConnectionClosed` resets the count, and `disconnect_peer_id` then closes the peer's other connections too.

→ `docs/invariants/network.md` § "A connection that just died is not a peer that stopped answering"

## A split keeps its conversation's prompt between turns (2026-10-09, #10)

**`pipeline::split_prompt_cache`** decides: a prompt pass carries a `PromptCacheHint` (keyed block chain, vLLM's rule; `resume_at` from an optimistic belief table) when EVERY remote segment advertises `features::SPLIT_PROMPT_CACHE`; wire trailers `0x0D` (forward, bound in the AAD) and `0x09` (blocks stored). The head gets TOKEN IDS from `resume_at`, never text. A miss is one retry from 0, never a penalty. A/B `SWARMLLM_SPLIT_PROMPT_CACHE=0`; rig `split_rig.sh splitcache`.

→ `docs/invariants/network.md` § "A split keeps its conversation's prompt between turns"

## A split reads its prompt in pieces, every machine at once (2026-10-10, #171)

**`pipeline::prompt_chunks`** runs a prompt pass of ≥ 2 pieces with one driver per segment, concurrently; a piece carries `LayerForward::prompt_span` (`0x0E`, AAD-bound, gated at the SENDER on `features::PROMPT_CHUNKS`), a peer's pieces are a stream with their own attempt tag. Never a machine twice; this node on EVERY boundary between segments unless the pass keeps its prompt (#10 — never chained, so it relays here whole anyway). A failure waits for the pieces out, then reads the pass whole once — ONLY if every piece was answered (`PiecesFailed::every_piece_answered`); a piece that may be live at a silent peer sends the error to the router. One piece to a peer until it answers one; a piece waits its turn `PIECE_TURN_WAIT` (600 s). A/B `SWARMLLM_PROMPT_CHUNKS=0`; rig `split_rig.sh splitcache` with `CPUS_A/B`.

→ `docs/invariants/network.md` § "A split reads its prompt in pieces"

## A tensor forward is acknowledged on receipt; a result is always its own request (2026-08-21)

`requests.rs` answers an inbound `LayerForward` with `SwarmResponse::Ack` on decode, BEFORE any work, from the network manager; a result is always its own request (`tensors::handle_send_tensor_result` → `send_tensor_result_as_request`). The fast-fail (`forward_ack_deadline_secs`) is gated ONLY on the peer's `features::FORWARD_ACK` bit and never reaps a slow answer — the compute deadline stays with the pipeline (gotcha #354). **A result says how long its node held the forward** (`LayerResult::served_us`, `0x0A`, the LAST trailer; #388): **`network::manager::served_time::ServedSince`** stamps the arrival on the network task at decode, in a bounded share per SENDING peer (the authenticated PeerId, never an address — gotcha #888; a peer flooding forwards pushes out only its own), and is emptied as the answer goes to the transport; only the peer that sent the forward is told (a chain's coordinator is not), to the millisecond, and only a coordinator advertising `features::SERVED_TIME` (bit 36). A forward that arrived by relay is not stamped. It is a number the far chose — read only through `dsd_flow::believed_served_ms`, never as a fact.

→ `docs/invariants/network.md` § "A tensor forward is acknowledged on receipt; a result is always its own request"

## A split token crosses on the pipeline stream, keyed by (request, peer) (2026-09-27)

`inference.persistent_pipeline_stream` is measured faster but stays OFF (no receipt ACK, #133). Streams are keyed by **(request, peer)**. Measure per-message cost on a REAL link; run the failover rigs before any default flip.

→ `docs/invariants/network.md` § "A split token crosses on the pipeline stream, keyed by (request, peer)"

## A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted (2026-09-27)

Substreams negotiate with multistream-select V1Lazy (`SWARMLLM_SUBSTREAM_V1=1` restores V1 for an A/B). `PeerInfo::latency_ms` and ACK samples are COSTS; a reader that means DISTANCE goes through **`network::manager::physical_rtt_ms`** (`exchange_says_lan`, `observe_network_coord`) — never compare a raw sample to a distance constant.

→ `docs/invariants/network.md` § "A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted"

## A burst is paced by the link, not by a window it never fills (2026-10-20, #318)

Every QUIC connection runs quinn's BBR: **`network::transport::quic_congestion`** → the vendored libp2p-quic's ONE addition, `Config::congestion`. Cubic (quinn's default, upstream's) grows its window only while a send fills it, and quinn paces each send at window / round trip — a split's bursts never fill it, so each was spread over ~0.4 of a round trip, and a loss shrank it for good. Sender-side only: no feature bit. A/B `SWARMLLM_QUIC_CC=cubic`; probe `examples/quic_pacing_probe`; a rig measures QUIC only with `RIG_TRANSPORT=quic` (loopback's 64 KB packets also hide TCP's slow start after idle).

→ `docs/invariants/network.md` § "A burst is paced by the link, not by a window it never fills"

## A speculative verify is walked where the logits are (2026-09-27)

A verify's last segment walks the drafts and answers with token ids (`LayerForward::spec_walk_at_tail`, gated at the SENDER on `features::SPEC_WALK_AT_TAIL`). **`sampling::sampled_accept_reject`** is the one rule; **`pipeline::VerifyReply::accept`** the one place a reply becomes accepted tokens; **`layer_forward::spec_trailer_flags`** the one writer of the flags byte. Shared noise (`coupling_seed`, `0x0B`, `features::COUPLED_SAMPLING`) is keyed by ABSOLUTE position, never relative (`a_drafter_keyed_at_the_samplers_positions_is_accepted_and_one_off_is_not`).

→ `docs/invariants/network.md` § "A speculative verify is walked where the logits are"

**A tree that FLOWS is drawn at the tail, never walked there (2026-10-22, #134 (b)).** A forward carrying `draft_tree_from` (`0x11`, after `0x10`, AAD-bound; gated at the SENDER on `features::DRAFT_TREE_FLOW` for every peer of the run — a hop's onward forward by `onward_refusal` + `draft_tree_from_onward`) appends its rows to the tree the segment's cache keeps since `from`, and the tail answers ONE draw per row (`sampling::draw_each_row`); the walk is the coordinator's. **`model_worker::draft_tree_to_apply`** is the one reading (refused unless `from ≤ index_pos` and the tree is `(index_pos − from) + positions` rows; `rows_for` checks it against the REAL cache); a sampler reading history is refused (`refuse_draws_with_penalties`).

→ `docs/invariants/scheduling.md` § "A tree that flows"

## A check travels a chain like a decode step does (2026-10-02)

**`PipelineExecutor::verify_may_chain`** decides whether `forward_verify_through_segments` sends a run of remote segments ONE forward (every hop advertising `features::CHAINED_VERIFY`, gated at the coordinator); a chained check resends on nothing (`ResendOnRefusal::Never`). **A chained run that fails is re-run UNCHAINED there** (segments cut back to `index_pos`) and sets the REQUIRED `chain_failed` — every caller passes `&mut self.chaining_disabled` (#318). A middle hop refuses what it cannot hand on BEFORE its worker (`layer_forward::onward_refusal`) and dials an unmet next hop (`NetworkCommand::DialPeer`, via `dial_checked`, its addresses — a DHT answer's too, its private ones dropped — through `filter_dialable`). **The sender names that target, so the dial is rationed** (`dial_next_hop`): `state.next_hop_dial_budget` (4/min per sender, 16 in all), none on a metered hold, a target the DHT did not know not looked up again for 60 s (`NextHopMisses`). `SWARMLLM_CHAIN_VERIFY=0` is the control arm.

→ `docs/invariants/network.md` § "A check travels a chain like a decode step does"

## A result names the step it answers; one that did not arrive is sent again (2026-09-25)

`LayerResult::answers_step` (`0x07`) names the forward — a serving node builds it from **`LayerForward::step`**; `PendingLayerResult::expects_step` refuses any other, and every registration sets it from the forward it actually sends. Only then may a serving node resend a lost result (`resend_lost_result`, gated on `features::RESULT_STEP`). Test with `SWARMLLM_FAULT_RESULT=lose|duplicate`.

→ `docs/invariants/network.md` § "A result names the step it answers"

## A stream of verifies runs in its order, and its answers are found by number (2026-09-30)

Streamed verify forwards (`LayerForward::stream_seq`, `0x0C`, gated on `features::STREAMED_VERIFY`) run in number order via **`daemon::state::forward_streams`**; `pending_layer_results` is keyed by `WaiterKey` and an answer echoes its number (`0x08`). A stream is ONE piece of its sender's work (`dispatch::StreamWorkSlot`, `MAX_STREAM_CHUNKS_HERE`); every admission refusal goes through `refuse_forward`. Stream only to a peer advertising `features::STREAM_AS_ONE_WORK`. **`dsd_stream::stream_shape`** takes any plan whose peers' segments are ONE run (the boomerang too, #152): where this node holds the last layers the walk is ours and those segments run as each answer is TAKEN, never as it arrives (#180). A run of several peers streams as a CHAIN (#318) only when every hop advertises `features::CHAINED_STREAM` (bit 27) and each machine appears once: a hop hands the number on (`layer_forward`), keeps a failure record per check (`hop_forwards`), and a restart goes once nothing runs, never waiting for a superseded turn (a hop before may have skipped it). A chained stream that fails awaits every chunk still out, then `dsd` finishes the reply in rounds, unchained (`EngineDrafter::follow`). **A check walked HERE that refused a guess skips the turns built on it** before the restart arrives (`Turn::refused_a_guess` ← `layer_forward::walk_refusal`; the restart clears it; A/B `SWARMLLM_STREAM_TAIL_SKIP=0`) — **a wrong look-ahead too**, where the coordinator names it (`LayerForward::stream_lookahead`, `0x12`, AAD-bound, gated at the SENDER on `features::STREAM_LOOKAHEAD`, bit 34, a lone far segment that walks: `dsd_stream::lookahead_to_send`; A/B `SWARMLLM_STREAM_LOOKAHEAD=0`, #382). **A likely refusal's restart is sent BEFORE the refusal** (`LayerForward::stream_branch`, `0x13`, AAD-bound, gated at the SENDER on `features::STREAM_SPARES`, bit 35, same peers as the look-ahead, greedy only, #381): a `Spare` runs only on the exact refusal it names (check, kept count, token) and only once, else it is skipped unrun; after one ran, the stale line is skipped WHATEVER the tail-skip switch (`forward_streams::Order::stood_on`), and an `AfterSpare` chunk runs only behind the spare it names. A spare is built AFTER its chunk is sent (`send_spare`). **Which spares go is PRICED, not floored** (`dsd_stream::spare_pays`: chance it runs × cycle saved − delay to earlier answers and the next chunk, averaged over this reply's last 16 arrival errors, `StreamRun::lateness`; A/B `SWARMLLM_STREAM_SPARE_LATENESS=0` = point price, `SWARMLLM_STREAM_SPARE_MIN` = the old fixed floor). **The coordinator is OFF by default** until measured on the 70B (opt-in `SWARMLLM_STREAM_SPARES=1`); the far end's support is always on (v0.3.264+). **A chunk is drafted with no confidence floor and every chunk is the stream's width, the last too** (`engine_drafter::stream_confidence_floor` — the rounds keep 0.4, and `draft()`'s `stop_below` is REQUIRED; `dsd_stream::chunk_guesses`, bounded by `StreamRun::positions_served`): a chunk cut short holds the stream a round trip (`held_by`, +18% without it TH↔AU), and a narrow tail is a new card graph late in a worker's life (10-59 s stalls, #883). A/B `SWARMLLM_STREAM_DRAFT_CONFIDENCE=0.4`, `SWARMLLM_STREAM_FULL_CHUNKS=0`. → § "A stream of checks travels a chain"

→ `docs/invariants/network.md` § "A stream of verifies runs in its order, and each answer names its number"

## Every reply a serving node sends goes to whoever is WAITING — failures included

**`layer_forward::reply_target`** is the single answer to who is WAITING; it returns a `ReplyTo` nothing else can build, so `send_error_result` cannot be handed the raw sender (the previous hop in a chain). Gotcha #707. A refusal that cannot be addressed past the hop it came from (a forward whose seal did not open) is PASSED ON by that hop: every `LayerResult` this node takes in goes through **`NetworkManager::dispatch_layer_result`**, which asks **`hop_forwards::HopForwards::take_onward_failure`** (an error, from the onward peer, about the step handed on, once) and reports it to the coordinator in its own name; `fail_tensor_forward` takes the forward's step as a REQUIRED argument (#305).

→ `docs/invariants/network.md` § "A chained hop's refusal goes to the coordinator"

## Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind (2026-09-26)

Every kind of peer work takes a `PeerWorkSlot`, and a refusal is ANSWERED — `layer_forward::refuse_forward` / `remote_generate::refuse_request`, worded by `peer_work_refusal()` — never silently dropped. A request that has not started leaves the last quarter of the slots alone (`admits_a_new_request`). Image-encode refusals still cannot be said (FUTURE_WORK #123).

→ `docs/invariants/network.md` § "Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind"

## UPnP that the router refuses is said out loud, and a stopping node hands its ports back (2026-10-08)

UPnP is the DIRECT `libp2p-upnp` ≥ 0.6 dependency (it backs off a refused mapping; the facade's 0.5.0 asked again without pause, for ever) — the facade's `upnp` feature stays OFF; guard `upnp_is_the_release_that_backs_off`. A refused mapping emits no event, so **`network::manager::upnp_watch::UpnpWatch`** explains the silence once (`UPNP_QUIET_AFTER`), and `release_upnp_mappings` closes the P2P listeners at a clean shutdown so the router gets the ports back (#239). It is the PATCHED copy in `vendor/libp2p-upnp` (igd-next 0.18): igd-next 0.16 could not reach a router that lists an absolute control URL (guard `upnp_reaches_a_router_that_lists_an_absolute_control_url`). **The external port is CHOSEN, never assumed (2026-10-24):** `vendor/libp2p-upnp/src/choice.rs` reads the router's table and never asks for a port another computer holds — a router may REPLACE another client's mapping without an error; the held port is what a renewal keeps and a removal deletes, and only while the table says it is ours. A read of the table goes past an empty slot (a router may answer 713 for one, the same code as the end, gotcha #950), and a read the router did not ANSWER is `Table::Unknown` — the request fails, never "empty" or "will not list". Only the listen address facing the router is mapped, one per listener; its other addresses wait in `waiting` and one is mapped when its mapping ends (`map_a_successor` — a new address can arrive before the old one goes). A reply from the gateway is checked against the listener's CURRENT address and an address that went is `Removing` until its own reply — never an `expect` on the mapping table (libp2p #6377 panicked the network task on an interface flap). A gateway not found is searched for again (2 → 30 min, #329). Tests: the crate's own (`vendored.yml`).

→ `docs/invariants/network.md` § "UPnP that the router refuses is said out loud"

## "Is this connection direct?" — `network::relay::addr_is_direct_transport`

**`network::relay::addr_is_direct_transport`** is the single answer, for both layers that choose a connection: a real `/ip4`, `/ip6` or `/dns*` hop AND no circuit. `is_relay_circuit_addr` alone misses the listener's view; the vendored request-response inbound handler records the send-back address. `peer_direct_conns` is gated on it; any new "prefer direct" logic uses it (gotchas #356, #179). Keep the loop-stall tripwire in `NetworkManager::run`.

→ `docs/invariants/network.md` § ""Is this connection direct?" — `network::relay::addr_is_direct_transport`"

## A relayed body is its origin's only once its origin is PROVEN

An ephemeral seal is anonymous (`crypto::relay_seal`): `origin` in the AAD is a claim anyone can make. A relayed message or tensor is dispatched as `Some(origin)` only when **`relay_seal::Opened::Proven`** (the origin's signature inside the seal), **`RelayReplayGuard::admit`** takes it (refuse when full, never forget early), and an envelope's inner is **`is_relay_eligible`**. Nothing is learned from a body before that (`take_proven_relayed_origin` — route, registry entry, proven features). A node relays ONLY to a target that `target_speaks(RELAY_SIGNED)`, always signed: an older target's reply comes back unsigned (`a_peer_that_cannot_sign_back_is_never_relayed_to`). A refusal goes through `refuse_relayed` (throttled per delivering peer). A new relayed kind gets its own `RelayedKind`, signed. Tests `a_relayed_{envelope,tensor}_is_dispatched_only_when_its_origin_is_proven` (#298).

**A share is counted by NETWORK, never by key or by a name the peer chose (#299):** the replay memory's per-deliverer share is the delivering peer's **`network::helpers::NetGroup`** — public IPv4 /16, IPv6 /32, a private host, a host name only if `ConfiguredNames` (our bootstrap list) has it, else one `Unaddressed` group; a circuit is `Circuit` of the network of OUR latest DIRECT connection to its relay (`manager::relay_network`, `PeerRemote::direct`; entries filed only by `record_peer_remote`), NEVER the relay's own share (the anchor relays for every NAT'd node) and never an address written in the circuit's, which libp2p-relay does not dial — recorded with the connection (`PeerRemote`) and applied in **`admit_relayed`**, the one admission. A proven origin's registry entry goes through **`identify::evict_one_registered_peer`** / `unconnected_registry_entries` (an unconnected entry before any connected peer, a BLANK one before one a capability landed in — #403's review: minted origins pushed out an honest relay-tier entry; at most `MAX_UNCONNECTED_REGISTERED` for relayed origins; every proven body marks its origin seen) and its mapping through **`remember_peer_id`** — never a bare insert.

→ `docs/invariants/network.md` § "A relayed body's origin is signed"

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
`PeerCondition::DisconnectedAndNotDialing`. **Peer exchange too**
(`handle_pex_response` → `discovery::group_dials_by_peer`): dialled one
address at a time, the first listed decided TCP or QUIC for the connection's
life. A peer's stored addresses are ranked by **`helpers::addresses_worth_passing_on`**
(public, then private, then this host; QUIC first), never cut in advertised
order — the cap dropped a 12-address peer's public QUIC address at random
(2026-10-23: fresh nodes met the Melbourne peer over TCP 6 of 6; fixed, QUIC 4 of 4).

→ `docs/invariants/network.md` § "One dial per PEER"

## An address another computer answered at is forgotten for the peer it named (2026-11-02, #155)

A dial that fails `WrongPeerId` proves the (peer, address) pair wrong: **`NetworkManager::forget_address_answered_by_another`** (from `dht::pair_proven_wrong`, in the `OutgoingConnectionError` arm) removes exactly that pair from Kademlia's table and the peer registry — and from the peer cache, which merges its stored entries into every save (`stale_peers::ProvenWrong`, read by `dialable_peer_cache` and `CacheWriter::save`, #413). libp2p-kad 0.48 forgets only after a transport error, never a last address — stale identities were dialled 3,148 times in 4 h. Never back off a PEER on it: whoever answers at an address can name any peer.

A peer evicts only on its OWN unreachability: out of reach for `stale_peers::UNREACHABLE_FOR` (since its last connection closed, or the node started — `LastReached`), its next failed dial (a transport error or `WrongPeerId`, never `Aborted`/`Denied`) takes it out of Kademlia's table (`leave_the_table_if_out_of_reach`) and `dial_bootstrap_peers` stops seeding it; Identify puts a connected peer back (#413: dead identities in every node's table, handed to every lookup).

→ `docs/invariants/network.md` § "An address another computer answered at is forgotten"

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
That matters because a provider record outlives the fact it asserts: this node
keeps one for 12 h and republishes every 4 h (`network::behaviour::KAD_*`, #91
(2f); it was 1 h / 20 min), and other peers serve it, so a node that
deleted or lost a shard is still advertised as holding it for hours. An
add-only writer wins every disagreement with a writer that removes, and its
cadence decides how fast.

→ `docs/invariants/network.md`

## A report built to be handed to a stranger is a publishing surface (2026-09-01)

**`network::redact::redact_addresses`** hides every host in the diagnostics report at the one point `api::admin::diagnostics` returns; `?full=1` is the opt-in. Guard: `the_diagnostics_report_hides_addresses_unless_full_is_asked_for`.

→ `docs/invariants/network.md` § "A report built to be handed to a stranger is a publishing surface"

## A peer's question is weighed, never granted — nothing is held or keyed by the asker's request id (2026-10-18, #252 (b))

`SwarmMessage::PromptAdmissionAsk` / `PromptAdmissionAnswer`, gated at the SENDER on `features::PROMPT_ADMISSION` (bit 25). The coordinator's wait is keyed `(ask_id, NodeId)` — a fresh id per QUESTION, since a re-plan keeps the request id (`pending_prompt_admissions`) — and dispatch resolves only the AUTHENTICATED sender's key; both ends resolve the peer with `resolve_connected_peer_id_bytes`. The asked node drops a malformed ask (`ask_is_well_formed`), answers a metered hold with `Refused`, rations weighing (`AskBudget`: 12/min per peer, 60/min in all — past it "go ahead" unweighed, by `try_send` from the dispatcher, no task) and weighs under a FRESH correlation id (`weigh_peer_prompt`): a request id is known to every machine on a route, so anything keyed or held by it is a peer's lever on a live request (the first design's two HIGH findings). The coordinator gives a refusal ONE class (`ServiceUnavailable`) and repeats its words through `peer_words`. A missing answer is "go ahead", never a refusal. A refusal says what its `layers_that_fit` counts (`Refused::counted`, `#[serde(default)]` — an older coordinator ignores it; a NEW `LayersCounted` variant needs a feature bit, or an older one fails to decode the answer). Worker side: `arch-worker-memory.md`.

→ `docs/invariants/memory.md` § "A split's peers are asked about room before the pass too"

## A request someone waits on is sent until answered (2026-10-15)

A gossip publish that succeeded says nothing about who received it. A pool join is resent (`pool::manager::JoinResend`) until accepted or its window ends — said out loud; the owner answers a repeat from the code's claimer with the SAME invitation (`PoolInviteCode::claimed_by`), at most once per `REPEAT_JOIN_ANSWER_GAP`. The CLI reports the outcome, not the send.

→ `docs/invariants/network.md` § "A request someone is waiting on is sent until it is answered"

## A metered connection holds back what this node uploads for others (2026-10-15, #277)

**`SharedState::metered_hold`** answers for uploads too; the network and dispatch loops and the capability read **`SharedState::metered_hold_now`** and never ask the system (the health monitor's tick refreshes it off-loop). While it holds: the capability withdraws inference (`can_serve_inference`), a new request that comes anyway is refused out loud through `dispatch::admits_a_new_request` (its metered answer REQUIRED; `ServiceUnavailable`, so the coordinator re-plans without a penalty), parts and prompt-cache snapshots are answered as a miss. Work already running, the owner's requests and gossip go on. No LAN exemption: `is_lan_peer` cannot tell a peer over a VPN. Guard: `every_upload_a_peer_starts_asks_whether_the_connection_is_metered`.

**Relaying**: **`SharedState::relay_forwarding_enabled`** is the one answer (read live; the hold stops the AUTOMATIC donation only). It reaches the server itself — `network::relay::OnlyWhileRelaying`, a required `build_behaviour` argument, refuses every reservation and circuit while it is false — and its DHT record is withdrawn on the falling edge (`discovery::stop_providing_relay_service`). Gotcha #861.

→ `docs/invariants/network.md` § "A metered connection holds back what this node uploads for others"

## A computer that shares nothing may look at the swarm, not use it (2026-11-04)

**`SharedState::swarm_use_of_peer`** (the dispatcher, for every request a peer STARTS here — inside `admits_a_new_request`, its verdict REQUIRED, and a tensor-parallel layer's prompt pass; a forward is judged as the coordinator it names, never the hop that chained it here — **`layer_forward::judged_as`**, asked through **`layer_forward::swarm_use_of_forward`**: a chained pass naming a coordinator with nothing on record here takes the authenticated sender's word where the sender advertises `features::JUDGES_THE_COORDINATOR` (bit 38) — it judged it by the same rule; a refusal on record stands) and **`swarm_use_here`** (`pool::scope::request_scope` → self + pool only; the scheduler's up-front `TooFewPartsToUseTheSwarm` / `TooFarBehindToUseTheSwarm`) are the one rule, `swarm_use::verdict`: at least `PARTS_TO_USE_THE_SWARM` (2) parts announced AND fewer than `RELEASES_BEHIND_TO_LOSE_USE` (2) releases behind the newest THIS node's update check found SIGNED (`UpdateInfo::signature_verified` — a bare tag never; a new minor version is ONE release, not infinitely far) — never a peer's claim (`PeerVersionWatch`'s rule). Judged from what the peer says of itself, so a node on an older release is judged too: its capability's version, and the MORE of its capability's parts and the parts its announcements credit it; this node by its last announcement (`own_shard_announce`, what peers credit). A capability naming an older version than the one held is never installed (**`swarm_use::capability_may_replace`**, the dispatcher — the one installer) — a replayed frame would otherwise get an honest peer refused — nor, on GOSSIP, one of the SAME version not built later than the held one (`NodeCapability::issued_at_ms`, `#[serde(default)]`, two unstamped ones as before; never news: `capability_news` carries it over; kept at most 5 min past our clock, `swarm_use::issued_at_as_kept` in `capability_as_kept`); one sent DIRECT is the peer's own word and is taken whatever its stamp (a clock set back); a refused one's latency readings are not merged (#402). No capability yet → allowed only while CONNECTED (it is on its way); not connected — a relayed origin, a computer cut off here (its entry went with the connection) — it is `NotHeardFrom` until its gossiped capability lands (`swarm_use_of_peer`'s `unheard`, #403); a pool device → allowed; no LAN exemption (the metered rule's reason). The refusal is this node's OWNER's: never raised for a split led here for a peer, and a head's hand-off failure sends it as plain "cannot serve" (`worker_failure_for_the_coordinator`). A refusal is `ServiceUnavailable`-worded (`SwarmUse::peer_refusal`): the coordinator bars and re-plans, no penalty. The PARTS half withholds only use — parts and gossip keep flowing; the VERSION half costs connections too (next rule). Behind is counted in PUBLISHED releases (`ReleasesKnown::behind`, the update check's list, which only lowers the count of numbers); a node with no list judges nobody behind. Display: `swarm_use_json` on all three stats surfaces, `NodeCapability::{auto_manage, updates_itself}` (display only).

→ `docs/invariants/network.md` § "A computer that shares nothing may look at the swarm, not use it", § "A computer this node has not heard from is not judged allowed"

## A computer two releases behind loses contact, not only use (2026-11-04)

**`SharedState::cut_off_for`** is the one judgment (the use gate's version half; never a pool device, never a computer this node bootstraps from — `SharedState::bootstraps_from`, its `/p2p/` id in the live `bootstrap_peers`, Bitcoin Core's `noban` — nobody while the pool state is busy) and **`NetworkManager::cut_off_peer`** the one actor, on the network loop: at Identify, before registering or catching up, when the agent version names a release (`swarm_use::agent_version`, `swarmllm/<version>` from v0.3.271); else on the liveness tick (`cut_off_peers_too_far_behind`) over **`peers_own_versions`** — only what the computer stated ITSELF (Identify, or a capability sent DIRECT while connected: `note_capability_heard`), never a gossiped or DHT one, which can be a recording. The cut blocks it (`allow_block_list`), drops it from Kademlia, and `dial_checked` / `try_enqueue_redial` skip it, for `CUT_OFF_FOR` (10 min), at most `MAX_CUT_OFF` (1024) blocked at once (keys are free; past it, only disconnected); then it is judged again. A relayed body whose PROVEN origin is cut off now is refused before dispatch (`NetworkManager::origin_is_cut_off`: after the proof, before the replay memory and the registry). One release behind keeps everything (no split at a release); updates come from GitHub, so a cut computer still updates. Bitcoin Core's `MIN_PEER_PROTO_VERSION` and EIP-2124 decide at the handshake too.

→ `docs/invariants/network.md` § "A computer two releases behind loses contact, not only use"

## A peer's catch-up that lands before its Identify is held, not dropped (2026-11-06, #415)

A message carries an authenticated sender only once the Identify handler has mapped its peer, and nothing orders Identify against request-response. **`NetworkManager::dispatch_authenticated`** (the one choke point) HOLDS a catch-up kind from an unmapped peer — `NodeCapabilityUpdate` within `dispatch::capability_fits`, `ShardAnnounce` within `fits_one_message`, `NicknameGossip` within `validate_nickname`; 8 messages and 1 MiB a peer, 8 MiB in all, 32 peers and 4 a `NetGroup`, 10 s (`network::manager::before_identify`) — and the Identify handler hands them on in order once the peer is mapped, registered and connected; a peer refused at Identify, or gone, has them dropped. Before, they went with no sender and were dropped: a restarted node lost a peer's parts list until its next repeat, and a pre-.271 computer was never cut off. A new kind a peer sends as it identifies us joins `is_a_catch_up`; never hold a forward or anything unbounded.

→ `docs/invariants/network.md` § "A peer's catch-up that lands before its Identify is held, not dropped"

## Every part this node serves shares ONE upload limit (2026-10-24, #335)

**`network::manager::upload_pacer::UploadPacer`** is the one timeline: an answer books its share of `shard_upload_mbps` (read live) BEFORE the read, for the bytes the part HAS past the offset (`bytes_to_book` — booked for the bytes asked, a request past the end held the limit and sent nothing), and is sent when its share ends — never a sleep per request (N parts at once went up at N times the limit). An answer holds at most 4 s of the limit (fewer bytes than asked; a downloader walks by bytes received) and is released within `RELEASED_WITHIN` (two thirds of the downloader's `SHARD_STALL_SECS`). Past that, and when the inbound queue is full, **`answer_part_request_busy`** answers BUSY to a downloader advertising `features::SHARD_BUSY` (bit 32) — it resumes at its offset (`ask_again_after_busy`, `next_ask`), never from 0 — and empty to an older one. A shared limit queues: bound every wait it adds by the requester's patience. **No network holds more of the queue than a node fetching at the default concurrency books** (`NETWORK_HOLDS_AT_MOST` = `DEFAULT_MAX_CONCURRENT_DOWNLOADS` × `ANSWER_HOLDS_AT_MOST`, 12 of 20 s, #340): each share is booked with the requester's network (`PeerRemote::group`, the `NetGroup` every share here is counted by — a key costs nothing to make), and past it the answer is busy (`Grant::Busy::network_share`) for the wait until its own oldest share ends; such a node is never bounded (half the window starved its third part). A downloader's part restarts its count of busy answers whenever the holder it waits on lands another of its parts (`note_chunk_landed_from`): that holder is serving it, not staying busy. An answer of no bytes goes at once, behind no share. A/B `SWARMLLM_UPLOAD_SHARE=0`; rig `split_rig.sh fetch` with `FETCH_OTHER`.

→ `docs/invariants/network.md` § "Every part this node serves shares ONE upload limit"

## Centralised Wire-Format Helpers

Single sources of truth for invariants that silently break at the wire if duplicated. Read the evidence before changing one.

- **`network::protocol::build_layer_forward_aad`** — ⚠ a new trailer is NOT a no-op for an older peer: feature-gate every optional trailer at the SENDER, and bind it in this helper (encrypt and `decode_layer_forward_encrypted` both go through it); clamp a clamped trailer AFTER the AAD is rebuilt.
- **`network::pipeline_stream::chunk_layer_forward`** — the one splitter for chunked sends.
- **`SharedState::resolve_pending_layer_result`** — the ONLY way to deliver a `LayerResult` into `pending_layer_results`.
- **`daemon::dispatch::timestamp_fresh_one_sided`** — the staleness primitive; `gossip_timestamp_fresh` wraps it.
- **`credit::ledger::check_signed_freshness`** — staleness for signed `DateTime<Utc>` messages.
- **`pipeline::pack_verify_tokens_to_le_bytes`**, **`pipeline::build_spec_verify_forward`**, **`pipeline::build_kv_truncate_forward`**, **`pipeline::register_pending_layer_result`** — the shared builders for verify/truncate forwards and waiter registration.
- **`storage::Database::with_write_table`** — the write-transaction wrapper; it hands the worker off before it blocks (#314).
- **`swarmllm_types::ShardResponse::empty()`**, **`ShardResponse::busy(retry_after_ms)`** (only to a `SHARD_BUSY` downloader), **`swarmllm_types::LayerResult::error(request_id, reason)`** — canonical refusal/error replies.
- **`network/manager/connections::try_enqueue_redial`** — dedup + cap + push for `pending_redial`.
- **`responses::types::raw_tool_kind_or_unknown`**, **`cli::bail_if_no_api_key` / `cli::exit_daemon_unreachable`** — shared error wording.
- **`model::auto_manage::spawn_check_and_load`** — "shard landed → reload model → refresh dashboard", always the three steps together.
- **`pool::invite::{encode_invite_code, decode_invite_code}`** — the `swarmpool://` v2 codec; `looks_like_v2` routes v2 from the legacy path. **`pool::invite::addresses_for_invite`** is what a code carries: the outbound route's own address (`outbound_addresses`) + internet-reachable ones — never a Docker bridge or an overlay off the default route (field report 2026-10-15); a route out by CGNAT (a Tailscale exit node) keeps the private LAN too.

→ `docs/invariants/network.md` § "Centralised Wire-Format Helpers"

## The models peers name are bounded, and one in good standing is never displaced (2026-10-24, #333)

**`SharedState::admit_a_peers_manifest`** (`daemon::state::peer_models`), asked LAST in `judge_peer_manifest` (`PeerManifest::NoRoom`), is the one bound: 1,024 models adopted as new from peers, 64 MiB of their stored JSON (`weight_of` — strings count; rows did not), 1 MiB a model; a model known before a peer named it is not weighed. Room comes only from a model NOBODY holds — **this node or a node CONNECTED to it** (`SharedState::holder_standing`; never `peer_registry`, where anyone can mint a relayed origin): a part announcement needs only a key, so a stranger's holder record keeps nothing — that this node has not acted on (`acted_on_model`), oldest first, 64 looks per ARRIVAL (CLOCK); none to spare refuses the newcomer (libp2p-kad `MaxRecords`, Bitcoin addrman's `IsTerrible`). The wishlist is not a protection (it is built from the registry). A forgotten model's stored row goes too, and `register_manifest` stores a peer's manifest only for a model with a part here. The door refuses an id past 256 bytes, a part number ≥ 256, a name past 512 bytes. Holder records: 4,096 per KEY (`MOST_PARTS_A_HOLDER`), not per computer — fresh keys are bounded by the next rule.

→ `docs/invariants/network.md` § "The models peers name are bounded"

## A full holder list takes a stranger's claim only in a stale record's place (2026-10-25, #339)

A part announcement needs only a key and a real far holder arrives the same way, so a node not connected here is displaced first and aged out, never refused. **`SharedState::holder_standing`** is where a holder stands — not connected, or connected from a network (`connected_networks`, a `NetGroup` key) — for the registry's hook AND #333's `holder_counts`; asked with NO registry guard held. **`holder_to_give_up`** at `MAX_HOLDERS_PER_SHARD`: a stranger takes only a stale (`STALE_HOLDER_AFTER`, 2.5 h) STRANGER's place, else it is ignored (libp2p-kad's `add_provider`); a connected node whose network already holds the most places takes its own network's NEWEST arrival's place, else the oldest stranger's, else the largest network's newest (Bitcoin's peer eviction — connections are cheap per host); a list that filled after the question ignores the claim. **`age_out_holders`** (every `FULL_REANNOUNCE_EVERY_TICKS` monitor ticks) forgets unconnected nodes' records past the bound, reverse index under the same guard. **A DHT provider record is the DHT's word, never the holder's** (#341: a `GetProviders` answer names any keys): `merge_dht_providers` writes `HolderRecord::first_hand = false` through the one writer `record_holder` — a stranger as claimant and as holder, not counted by `part_has_a_holder_where`, aged out whoever it names, refreshing no holder's own record; only the holder's own claim makes it first-hand.

→ `docs/invariants/network.md` § "A full holder list takes a stranger's claim only in a stale record's place"

## Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer

Publish on CHANGE, never per item per tick — gate on a digest of what is ASSERTED, excluding any timestamp. Gossip only what this node MEASURED (`local_region_demand`; guard `the_demand_we_gossip_is_the_demand_we_measured`) — and of that only work served FOR PEERS (`SharedState::note_request_served_for_a_peer`; the owner's own requests stay in `region_demand`, and the wishlist announces `announced_score`, without the demand term — #389 (a)). Catch a newcomer up POINT TO POINT (`NetworkCommand::SendDirectMessage`), and send our capability AND our parts list at identify time (`ModelMgmt::own_shard_announce`, kept by every health round; without it a restarted node planned splits blind for ~5 min, #359) — and our name while it is shown (`identity::nickname::own_record_for_a_newcomer`, signed afresh past 12 h; a rename is gossiped once, #417). Don't repeat what the swarm just HEARD (`state.models.manifest_heard`, `MANIFEST_QUIET_WINDOW`; only GOSSIPED arrivals count, keyed by VERSION) — but a version heard before this node last changed its OWN is not heard: a return to it goes out (`manifest_needs_broadcast`, #224 — a newcomer caught up mid-change never learned the agreed hash). Our capability and our parts list go out when they say something new, and an UNCHANGED one again at a wait that doubles while nothing changes — 5, 10, 20, then 40 min (**`health::monitor::QuietRepeat`**, RFC 6206's Trickle without suppression; a change resets it; the parts list never sooner than 10 rounds, so no gap passes the 50 min `STALE_HOLDER_AFTER` is three of — guard `no_repeat_waits_past_a_third_of_the_time_a_holder_record_goes_stale`; A/B `SWARMLLM_QUIET_REPEAT=fixed`, #91 2026-10-29). **`health::monitor::capability_news`** compares whole capabilities, carrying over only uptime, the latency hint, free RAM wherever its one reader (`memory_for_model_layers_mb`) would not see it, and figures inside their deadband (memory 256 MB, disk 1 GB, coordinate 5 ms, a worker's own decode timing 1.5x) — a new field is news by default; guards `an_idle_nodes_capability_is_not_news_round_after_round`, `free_ram_is_news_only_where_a_peer_reads_it`, `a_workers_own_speed_is_news_only_past_its_deadband`. Liveness is `HealthPing`'s job, never the capability's. **Gossip only what a reader uses**: `RegionShardSummary` was a third of an idle node's gossip and read by nothing, and is no longer built (guard `no_node_builds_a_region_summary_to_gossip`, #91). The mesh is sized from the PEERS the cache names (`discovery::peers_named`), never its address count: 200 addresses were 26 peers, and the 100+ tier doubled every message's copies (#91 (1b)). **A peer advertising `features::MANIFEST_DIGESTS` is told the VERSIONS (`ManifestDigest`) and pulls (`ManifestPull`) only what it lacks** — at its catch-up and every `MANIFEST_QUIET_WINDOW` (`health::monitor::send_manifest_digest`; receiver rules in `daemon::state::manifest_sync`, #91 (2)). **A new connection is a new catch-up** (BEP 3): `ManifestSync::note_new_connection` on every connection established numbers it and forgets which models the peer was answered (at most once per `ANSWER_AGAIN_AFTER`), and the round tells a digest-taking peer on a newer connection again — a per-peer memory that outlives the connection left a restarted node blind to the swarm's models for up to 2 h. The record goes with the LAST connection, keyed by `peer_id_to_node_id` as it was filed. A CHANGE still goes on gossip in full; the gossiped repeat waits `MANIFEST_QUIET_WINDOW_WITH_DIGESTS` (24 h) only while EVERY connected peer takes digests. Neither message goes on gossip, and one arriving there is not answered.

→ `docs/invariants/network.md` § "Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer"

## A parts list longer than a receiver takes goes in pieces, each model whole (2026-11-02)

Every node drops a `ShardAnnounce` past **`swarmllm_types::MAX_SHARDS_PER_ANNOUNCE`** (512) unread, so a sender never sends one: **`network::manager::commands::pieces_if_too_long`** — asked by the gossip publish and the direct send, never at the eleven call sites — cuts it with **`ShardAnnounce::into_pieces`**, each model's parts, tags and flags in ONE piece (a receiver retracts per message for a model listed complete). No wire change; never raise the bound (older receivers keep it). **Every list in it is bounded, by ONE predicate on both sides** — `ShardAnnounce::fits_one_message`: parts, `origin_checked_models` AND `complete_for_models` (2026-11-04: the last was read unbounded, a pass over the parts and a registry write per name); a new list in the message joins it. A 565-part holder was invisible to the whole swarm for 6 h without it.

→ `docs/invariants/network.md` § "A parts list longer than a receiver takes goes in pieces"

## A peer's clock may run minutes fast; gossip's gate allows 5 (2026-11-06, #416)

**`network::manager::events::gossip_out_of_time`** gates timed gossip (health pings, parts lists, credit gossip): at most `GOSSIP_MAX_AGE_SECS` (300) old and `GOSSIP_AHEAD_SECS` (300) ahead — credit gossip the ledger's `CLOCK_SKEW_TOLERANCE_SECS` (30), which `verify_balance_report` enforces anyway — one-sided (gotcha #44). At 30 s ahead a computer a week's clock drift fast (a standalone Windows PC syncs weekly) had every message dropped by every peer. A kind joins the gate only if it is kept per sender; the other timed checks (pool, nicknames, demand gossip, relayed bodies) guard a replay memory or a shared slot and are weighed one by one (FUTURE_WORK #416) — never one shared constant raised for all. The owner is told at 20 s off GitHub's `Date` (+ `Age`) header by the release check (`update::note_clock`, `CLOCK_OFF_TELL_SECS`) — never by peers' stamps, which they could move. A nickname's stamp is checked ONLY by
`NicknameRecordExt::verify` (5 min ahead, 24 h old; `dispatch::take_nickname_gossip`) — never a second,
tighter check in front of it; and a new record is stamped after the computer's previous one
(`new_signed`'s REQUIRED `after`): a rename is sent once.

→ `docs/invariants/network.md` § "A peer's clock may run minutes fast"

## A peer that reconnects beside an old connection is told our subscriptions again (2026-11-05)

Gossip runs on the VENDORED `libp2p-gossipsub` 0.49.5 (`vendor/libp2p-gossipsub`, `[patch.crates-io]`): **`Behaviour::send_our_subscriptions`** re-sends every topic we subscribe to when a connection opens beside another (`other_established > 0`) AND when one closes beside a live one — a peer's connections share ONE send queue, so the copy queued while a dead connection's handler ran can be lost into it. Upstream tells only the first connection; since 0.49.5 a node drops a GRAFT for a topic the peer never SUBSCRIBEd to, so a computer that restarted before our old connection timed out never heard us and its gossip reached nobody holding one — an updated AU node's capability (location, memory, residency) stayed at its first second on every peer for hours. Never drop the patch on a re-vendor; guard `gossip_runs_on_the_vendored_copy_that_tells_a_reconnected_peer_our_subscriptions`; the crate's own test runs in CI `vendored.yml`. A peer whose `uptime_seconds` does not advance between two reads is the symptom. **And a node closes every connection before it restarts into an update or stops** (`NetworkManager::close_every_connection`, #407): `exec` closes nothing, and a QUIC peer holds the dead connection until its idle timeout.

→ `docs/invariants/network.md` § "A peer that reconnects beside an old connection is told our subscriptions again"

## A computer's country is read as two letters, in one place (2026-11-05, #407)

**`api::admin::country_code`** reads every country — a peer's text (peer list, network map, swarm capacity, a route's regions, MCP, the identity card) and this node's own lookup: two ASCII letters, never `XX`. **A peer's region is read as it ARRIVES**: the capability installer goes through **`api::admin::capability_as_kept`**, so every registry reader (the scheduler's region score, auto-manage, the relay's records) sees two letters or nothing (guard `every_peer_capability_is_installed_with_its_region_read_as_a_country`); a new installer calls it. This node's own is `effective_region` / `_sync`: the LIVE `identity.region` through `country_code`, else the lookup (2026-11-06). **`api::admin::map_region_of`** places a peer on BOTH map paths (`network_map`, the WebSocket `region_summary`): its country, ours only for a LAN peer, else `??` — counted, drawn nowhere, never put in our country. The lookup asks ip-api.com, then Cloudflare's trace, again on a backoff (`background::region_retry_after`).

→ `docs/invariants/network.md` § "A peer that reconnects beside an old connection is told our subscriptions again"

## A counter named for an outcome may only be counting the attempt

GossipSub's `sent` / `sent_bytes` count ATTEMPTS per recipient, so `sum(topic.sent_bytes) <= out_bytes` must NOT be asserted — the gap is the signal. Read both drop paths (queue expiry via metrics; queue full only via `gossipsub::Event::SlowPeer`). `sent_msgs` is not a publish rate; `published_msgs` is. Before estimating a total's components, look for the counter you are not reading (gotcha #673).

→ `docs/invariants/network.md` § "A counter named for an outcome may only be counting the attempt"

## A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT

The per-shard tensor table is ~92% of a manifest and is derived, not sent: **`daemon::shard_loader::derive_tensor_entries`** rebuilds it as a FALLBACK. Every table — published or derived — is written by **`LayerShardLayout::tensor_entries`** (#334 (2)). ⚠ `manifest.shard_count` is the layout's OUTPUT, so the count is SEARCHED for with the published `size_bytes` as oracle; return `None` rather than a partial table; do NOT "fix" the underproduction. `a_derived_tensor_table_matches_the_published_one` (ignored; `SWARMLLM_TEST_MODEL_DIR`).

→ `docs/invariants/network.md` § "A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT"

## A request reaches only the nodes its scope names, on every path (2026-10-12)

**`pool::scope::request_scope(shared, model_id, asked)` is the single answer** to which nodes a request's data may reach — "Keep on this computer" by the owner (per model) or by the REQUEST (`KeepAsked`, #251 (2b): `api::keep_here` reads the header, the body's `swarmllm` field or MCP's `_meta`; the strictest wins, an unreadable one is a 400) → this node alone, else private mode's set; `allowed_node_set` is for shard management only. `InferenceRequest::local` takes the keep as a REQUIRED argument; a path that writes `KeepAsked::Unasked` for a request is listed in `KEEP_UNASKED_IS_THE_ANSWER` (guard `a_requests_keep_instruction_is_never_replaced_by_none`). A kept request's prompt-cache entries are never announced or exported (`prefix_cache::Entry::kept_here`, sticky). Guards `every_path_a_request_leaves_by_asks_its_scope`, `every_cloud_path_asks_whether_the_model_is_kept_here` (`providers::cloud_refused_for`). A failure is never broadcast to the swarm.

→ `docs/invariants/network.md` § "A request reaches only the nodes its scope names, on every path"

## Network coordinates: publish them rough, feed them a MINIMUM

`NodeCapability.coord` (Vivaldi, `features::NETWORK_COORDS`) is published however rough (`a_brand_new_node_still_publishes_its_coordinate`; `is_usable()` is the CONSUMER's gate). Feed the windowed MINIMUM, never a raw sample — **`SharedState::observe_network_coord`** is the single writer, `LatencyFilter` takes the minimum (not Serf's median). Size a time window against its fill rate: `LATENCY_WINDOW_MIN_SAMPLES`, `LATENCY_SAMPLE_MAX_AGE_MS`.

**Two coordinates (#91): the capability carries one that MOVES only when the node has moved, the timed answer the one as it stands.** **`netcoord::UpdateFilter`** (Ledlie et al.'s RELATIVE heuristic: centroids of a frozen start window and a sliding one, 32 each, change past 0.3 of the round trip to the nearest neighbour, floor `UPDATE_MIN_MOVE_MS`; whole windows decide WHEN, the newest quarter WHERE; the first full window after a change re-centres, and a node's first window counts as a change) is fed only by `observe_network_coord` and read only by `network_coord_for_publication`; a peer-exchange answer carries `network_coord_now` (`PeerExchangeResponse::coord`, Serf's `AckPayload`), preferred by the asker over the capability's (`observe_network_coord`'s REQUIRED `sent_with_it`). Never a fixed threshold on the raw coordinate: an orbit crossed 5 ms on 16 of 18 broadcasts. A coordinate that became usable (or stopped being) is news to the start-up filter and to `capability_news`, however little it moved. A restart re-converges; a coordinate restored across one was measured and NOT shipped (Serf's lesson too). A/B `SWARMLLM_COORD_FILTER=0`; DIAG `published network coordinate moved`.

→ `docs/invariants/network.md` § "A network coordinate must be published rough, and fed a MINIMUM"

## Single-source-of-truth helpers — Network protocol, peers and the model registry

Each names the ONE place a decision is made; a second implementation is this codebase's most-repeated defect (`.claude/rules/architecture.md` § "One invariant, N paths"). Read the topic file before changing one.

- **`SharedState::publicly_reachable`** — is this node reachable from the internet (a confirmed public external address). `node_stats.nat_status` is the last per-address NAT event, never the answer (`api::admin::nat_headline`, gotcha #796).
- **`inference::pipeline::remote_generate::StreamReassembler`** — puts a remote reply's token stream back in order.
- **A hole in a peer-served reply is FILLED, not waited out** — `RetainedReplies`, `SwarmMessage::ResendTokens`.
- **`NodeCapability.cpu`** — a processor described like a graphics card.
- **`PeerInfo::ack_srtt_ms` is what routing prices a peer by** — capped at `ACK_SRTT_ROUTING_CAP_MS`. **`PeerInfo::round_trip_ms`** is the one COST reading (it, else the ping) for routing and a wait; a DISTANCE is the windowed minimum (`metrics.network_coord_samples` → `LatencyFilter::min`: the coordinate, near chains, `trees_wanted`) — the smoothed cost read 2.5-5 s on a 20 ms link through a first request.
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
- **`pool::types::member_is_online`** — is a pool member on line: this device, connected now, or reported within 2.5 report intervals; never `PoolMembership::online` (set, never cleared — a device stopped two days showed `[on]`, field report 2026-10-15). Guard: `a_pool_members_liveness_is_never_read_from_its_stored_flag`.
- **`ModelManifest::context_length`** — the model's declared context, OUTSIDE `manifest_hash` (like `mmproj`; a hashed field would make old and new nodes disagree on every manifest); filled by `SharedState::fill_declared_contexts` before gossip, kept unknown → known by `register_manifest` (#189).

→ `docs/invariants/network.md` § "Single-source-of-truth helpers — Network protocol, peers and the model registry"
