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
  - "vendor/quinn-proto/src/**"
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

`SessionManager::remove_session` retires the live key (openable, never sealable, for `PREVIOUS_KEY_GRACE`) on EVERY full disconnect, mid-pipeline too. A failed `open` calls `request_rekey` (inside `open`, so every decrypt site inherits it), even with no session; the refused forward is resent once repaired (`ForwardRefusal::Undecryptable`, `0x06`, `features::FORWARD_REFUSAL_REASON`; `arch-scheduling.md`) and names its forward (`LayerForward::step`; #303). An age eviction retires the same way: `evict_stale` returns the peers, the rotation task re-keys each (#304).

→ `docs/invariants/network.md` § "A disconnect retires a session key; it must not destroy it"

## A key derived while ANSWERING an exchange waits until the peer proves it has it

A responder never installs a key it derived while answering an exchange until the peer proves it has it: `accept_ephemeral_exchange` parks it in `unconfirmed`, `open` promotes it. The initiator's `SESSION_CONFIRM_MARKER` comes from `complete_ephemeral_session` so seal and install cannot come apart. Gated on `features::SESSION_KEY_CONFIRM`; WireGuard's rule.

→ `docs/invariants/network.md` § "A key derived while ANSWERING an exchange waits until the peer proves it has it"

## A downloaded shard is hashed OFF the event loop

Never hash a whole shard on the swarm event loop (#108): hash on the blocking pool, post the verdict back (`shard_verdict_tx` → `requests.rs::finish_p2p_shard`, ON the loop). Guard `the_network_loop_never_hashes_a_shard_inline`.

→ `docs/invariants/network.md` § "A downloaded shard is hashed off the event loop"

## The network loop never waits for the disk (2026-10-20, #314)

`storage::Database::with_write_table` hands the worker's queue off before a commit; the loop's OWN writes go to the blocking pool (`peer_cache::CacheWriter`). Every arm starts with `loop_stall::ArmStart::now()` (set `loop_stall::ArmDetail` if it runs commands or messages); the swarm is polled through `loop_stall::TimedNext`. **No header parse on the loop** (Identify asks `SharedState::refused_model_if_learned`, never `refused_model` / `gguf_meta_for`; the health tick learns geometry) and **no guard on a map the loop touches (`peer_registry`) across a database write** — change under the lock, write after, many rows in ONE commit (`Database::put_json_many`).

→ `docs/invariants/network.md` § "The network loop never waits for the disk"

## One writer per shard file, and finishing one shard says nothing about the others

One writer per `shard_NNN.bin.tmp`: `ModelMgmt::claim_shard_download` (RAII), `SharedState::remove_acquisition_if_idle` (the only tidy-up removal of a progress entry), `ModelMgmt::live_cancel_flag` (the one cancel flag). Only a download deletes its partial file, so EVERY writer claims from its first byte — a peer transfer through `handle_send_shard_request`'s REQUIRED `PartAsk` (`ModelMgmt::hold_for_a_peer_transfer`, `p2p_transfer_still_owns_the_file`; #418), a staging name through `StagingFile` (#419). A transfer resumes (`PartTransfer`); a failed check blames only a SOLE sender (`PartTransfer::sole_sender`, #191).

→ `docs/invariants/network.md` § "One writer per shard file, and finishing one shard says nothing about the others"

## Destroying a shard we hold needs better evidence than a stranger's claim

**`ModelRegistry::mismatch_policy`** is the single answer to "may this node quarantine its own copy?" — only when `expected` is the hash from the model's ORIGIN; `ShardStore::verify_shard` takes `OnMismatch` as a required parameter. A gossiped hash is a claim about the claimant's build; a disagreement is kept and served, NOT settled (`docs/FUTURE_WORK.md` § "A disputed shard is kept but the disagreement is never settled"). The one deletion on peers' word is the heal on a node with no origin to ask (`settle_by_checked_holders`, below).

→ `docs/invariants/network.md` § "Destroying a shard we hold needs better evidence than a stranger's claim"

## A holder claim means VERIFIED, not transferred

Nothing records a peer — or this node — as holding a shard until its BLAKE3 check passed. `daemon::dispatch::progress_claims_the_peer_holds_it` is the one reading of a `ShardDownloadProgress` (needs `DownloadState::Complete`; a percentage never asserts holding, `IN_FLIGHT_MAX_PCT`; #78, #184). `SharedState::parts_servable_here` is the one answer to which parts can be served: credited, no download under way, file EXACTLY its manifest size (`shard::shard_file_is_whole`, never "within 10%", #334 (1)); its base `SharedState::parts_landed_here` is what acquisition reconciles against every 10 s (#420). A download credits only what it FETCHED (`settle_finished_download`; `manifest::generate_manifest_from_header` credits none, #423) and an explicit one asks first (`admin_hf::shards::parts_already_here`, #393).

→ `docs/invariants/network.md` § "A holder claim means VERIFIED, not transferred"

## The activity list reports a TRANSITION; the log may report every message

The 100-entry `emit_activity` ring is the dashboard replay and the report's recent activity: gate an ActivityEvent on a state CHANGE, one entry per event not per item; the per-message DIAG log stays.

→ `docs/invariants/network.md` § "The activity list reports a TRANSITION; the log may report every message"

## A holder record names a BUILD, not just a shard

**`ModelRegistry::shard_holders`** is the single read accessor and filters out holders of a different build (`expected_build_tag`); `all_shard_entries` is the RAW map — never show or decide on a holder count from it. ⚠ "Holds a part" (`peers_hosting`) and "could serve it alone" (`peers_complete`) are two counts. `ModelPeerCounts` carries `servable` and `other_build` together. Guard: `a_holder_count_shown_to_a_person_is_the_count_that_can_serve`.

→ `docs/invariants/network.md` § "A holder record names a BUILD, not just a shard"

## One upload per model id, on every node (2026-10-02)

`model::canonical` is the single answer to "which file IS this model": `hf_sources` is written only via `note_origin_claim` / `adopt_canonical_build`, a header only via `fetch_model_header`, every download asks `canonical_allows_acquisition`, `auto_manage::canonical` heals a node holding another upload (DELETES parts that are not the upload's bytes, refetches through the repair queue, only after HuggingFace answered that pass, never while the model is in use). ⚠ **Never reorder `TRUSTED_HF_PUBLISHERS`** (#151). A peer's manifest goes through `SharedState::judge_peer_manifest` (#158).

**A node vouches only for bytes that are the swarm's upload:** announced tags are the tags of the BYTES (`ModelRegistry::announced_build_tag`); verdicts reach the dispute only through `SharedState::note_shard_disputed` / `note_shard_verified` (`ModelRegistry::note_bytes_checked`); a pass never clears a dispute outright (#224). A copy waiting to be replaced is WITHHELD: `SharedState::note_canonical_holding` is the one writer of `canonical_holding` and `ModelRegistry::set_model_withheld`. A peer's part is byte-checked (`auto_manage::canonical::part_is_from`).

**No origin to ask → healed by holders that checked theirs (#160):** `ShardAnnounce::origin_checked_models`; `settle_by_checked_holders` removes a part only when `CHECKED_QUORUM` (2) CONNECTED checked holders agree and none holds ours. Where HuggingFace answers, the UPLOAD settles.

**A vision encoder is fetched only by its own origin, and pinned (#422):** `model::huggingface::vision_encoder::fetch_from_origin`, from this node's own HuggingFace downloads only (never a peer's manifest or gossip, #421); llama.cpp's file (`best_encoder_for`) or the pinned one by HuggingFace's listed SHA-256 (`listed_with_sha256`, never a name). The pin (`ModelManifest::mmproj`, wire `vision_encoder`: v0.3.276 acts on a `mmproj` key) is kept like a part's hash (`manifest::keep_known_vision_encoder`; `MAX_VISION_ENCODER_BYTES`); a peer's copy is checked (`manifest::expected_part`) or refused (`P2pShardAcceptance::Refuse`). → `docs/invariants/network.md` § "A vision encoder is fetched by its own origin, and pinned"

**An origin hash ends with the bytes it vouched for (#270):** `forget_origin_hashes_of_copies_gone_from_here`; the heal judges every directory under `models/` (`models_kept_on_disk`); never make the refusal itself ask about held parts (#384).

→ `docs/invariants/network.md` § "One upload per model id, on every node"

## ACK-Timeout Fast-Fail for rr Sends

Streaming rr sends MUST set `SendDirectMessage.delivery_request_id = Some(uuid)`; the `RR_ACK_TIMEOUT_SECS` sweep then closes `streaming_token_txs[uuid]` when libp2p drops a send silently. Pair with the `is_transient_remote_failure` retry in `dispatch_single`.

→ `docs/invariants/network.md` § "ACK-Timeout Fast-Fail for rr Sends"

## A connection the swarm DENIED is forgotten by request-response (2026-10-02)

**`forget_denied_connection`** drops a request-response connection the swarm denied (per-peer cap) and fails its requests. Anything keeping per-connection state must clean up on `ListenFailure` / `DialFailure` as well as on a close.

→ `docs/invariants/network.md` § "A connection the swarm DENIED is forgotten by request-response"

## A request failure closes a connection only when it means silence (2026-10-16, #290)

**`connections::failure_counts_as_silence`**, asked inside `note_rr_failure`: only `Timeout` (and `UnsupportedProtocols`) count — a transport-lost connection fails every in-flight request BEFORE `ConnectionClosed` resets the count, and `disconnect_peer_id` would close the peer's other connections too.

→ `docs/invariants/network.md` § "A connection that just died is not a peer that stopped answering"

## A split keeps its conversation's prompt between turns (2026-10-09, #10)

**`pipeline::split_prompt_cache`** decides: a prompt pass carries a `PromptCacheHint` (keyed block chain, vLLM's rule; `resume_at` from an optimistic belief table) when EVERY remote segment advertises `features::SPLIT_PROMPT_CACHE`; wire trailers `0x0D` (forward, bound in the AAD) and `0x09` (blocks stored). The head gets TOKEN IDS from `resume_at`, never text. A miss is one retry from 0, never a penalty. A/B `SWARMLLM_SPLIT_PROMPT_CACHE=0`; rig `split_rig.sh splitcache`.

→ `docs/invariants/network.md` § "A split keeps its conversation's prompt between turns"

## A split reads its prompt in pieces, every machine at once (2026-10-10, #171)

`pipeline::prompt_chunks` runs a prompt pass of ≥ 2 pieces with one driver per segment, concurrently; a piece carries `LayerForward::prompt_span` (`0x0E`, AAD-bound, gated at the SENDER on `features::PROMPT_CHUNKS`). Never a machine twice; this node on EVERY boundary unless the pass keeps its prompt (#10). A failure reads the pass whole once, ONLY if every piece was answered (`PiecesFailed::every_piece_answered`). One piece to a peer until it answers one; a piece waits `PIECE_TURN_WAIT` (600 s). Each piece is timed for its segment's prompt PRICE (`PassTimed::add`, `note_prompt_timed`). A/B `SWARMLLM_PROMPT_CHUNKS=0`; rig `split_rig.sh splitcache` with `CPUS_A/B`.

→ `docs/invariants/network.md` § "A split reads its prompt in pieces"

## A tensor forward is acknowledged on receipt; a result is always its own request (2026-08-21)

`requests.rs` answers an inbound `LayerForward` with `SwarmResponse::Ack` on decode, BEFORE any work; a result is always its own request (`send_tensor_result_as_request`). The fast-fail (`forward_ack_deadline_secs`) is gated ONLY on `features::FORWARD_ACK` and never reaps a slow answer (#354).

**A result says how long its node held the forward** (`LayerResult::served_us`, `0x0A`, after `0x09`; #388): `network::manager::served_time::ServedSince` stamps arrival at decode, per SENDING peer (authenticated PeerId, never an address, #888); only the forward's sender is told, to the millisecond, and only if it advertises `features::SERVED_TIME` (bit 36). The far's number: read only through `dsd_flow::believed_served_ms`. **And how it served it** (`LayerResult::how_served`, `0x0C` after `0x0B`, `features::HOW_SERVED` bit 43): `LayersFrom` (loaded / a copy's view, #449 / disk; the WHY only to `features::HOW_SERVED_WHY`, bit 44, else `LayersFrom::without_reason`) plus `worker_us`; `served_time::tell_the_recipient` is the ONE rule both ways an answer leaves. Logged, never priced.

→ `docs/invariants/network.md` § "A tensor forward is acknowledged on receipt; a result is always its own request"

## A chain hop tells its coordinator its own time, once, straight (2026-11-12, #427)

Each hop notes its held time — arrival (`ServedSince`, keyed by the predecessor) to its onward send or its answer to a coordinator that did not send it (`NetworkCommand::NoteHeldAsHop`) — for one-position decode steps, in `network::manager::hop_timings::HopHeld` (256 requests, 512 steps, 30 min), and answers the coordinator's own `CancelInference` (authenticated sender, never gossip) with `SwarmMessage::HopTimings` (its median), once, gated at the SENDER on `features::HOP_TIMINGS` (bit 40), never relayed (not `is_relay_eligible`). Each sending peer holds its own share (`MOST_REQUESTS_A_PEER`, 64). Never a list each hop appends on the forward (every hop re-seals it). Coordinator side: `arch-scheduling.md`.

→ `docs/invariants/network.md` § "A chain hop tells its coordinator its own time"

## A hand-off's prompt read is reported while it runs, and only lengthens a wait (2026-11-12, #430)

The serving peer reports `SwarmMessage::PromptProgress` where the request ASKED (`RemoteGenerateRequest::wants_prompt_progress`) and the requester advertises `features::PROMPT_PROGRESS` — at most every 2 s, only once it moved (`ProgressTargets`, ≤ 256). **A report is a share of the pass's own read (`OUT_OF` = 1,000), never a position**; `frame_due` is the one builder and leaves out what the prompt cache held. The requester keeps a report only from its attempt's `expected_peer`, only where it advances (`ProgressSeen`), and waits 120 s past the last (twice the longest gap where longer), never past an hour from the send (`remote_generate::first_token_deadline_after`). A report never shortens a wait or prices a peer; not relay-eligible. A split led by a peer is reported too (`ReadBy::Chain`); **a chain's pass that began past 0 is never reported**.

→ `docs/invariants/network.md` § "A hand-off's prompt read is reported while it runs"

## A hand-off's last frame says how its writer guessed (2026-11-14)

`GenerateUsage::how_guessed` (`HowGuessed`) goes worker → `GenerateDone` → `StreamingTokenTx::note_how_guessed` → the last frame; the requester logs it once (`remote_generate::log_how_the_writer_guessed`). Never WHY a reply was not guessed; `Speculated` only where a check RAN (batched, ruled out, "checked nothing" are `Plain`). ⚠ One bit ACCEPTED: `Speculated` says no other request was decoding on that worker on arrival. Never add a field that narrows `Plain`. Counts and fixed words only; unknown words `Unknown`, an unreadable record `None`; the peer's word, logged, never priced.

→ `docs/invariants/network.md` § "A hand-off's last frame says how its writer guessed"

## A block drafter's rows come back with the answer, asked for and bounded (2026-11-08, #428 stage 3)

A forward to a segment running a layer the block drafter reads carries `LayerForward::block_rows_wanted` (`0x14`, after every other trailer, AAD-bound: layers ≤ `MOST_BLOCK_ROW_LAYERS`, how many last rows at most), gated at the SENDER on `features::SENDS_BLOCK_ROWS` (bit 39); only `PipelineExecutor::block_rows_wanted_for` builds one. The answer's `LayerResult::block_rows` (`0x0B`, after `0x0A`) holds bf16 rows of only the asked layers the segment runs, bounded on read by `MOST_BLOCK_ROW_BYTES`; a cut-short, empty or over-bound trailer is refused, never read as absent. The coordinator holds each answer to what it asked of THAT segment (`BlockRowsAsk::note`, REQUIRED segment and rows) and again before the drafter reads it (`block_rows_for_the_drafter`; `PROMPT_WINDOW`). The serving side never holds more on its device than the ask allows; an onward hop drops the ask.

→ `docs/invariants/network.md` § "A block drafter's rows come back with the answer"

## A split token crosses on the pipeline stream, keyed by (request, peer) (2026-09-27)

`inference.persistent_pipeline_stream` is measured faster but stays OFF (no receipt ACK, #133). Streams are keyed by **(request, peer)**. Measure per-message cost on a REAL link; run the failover rigs before any default flip.

→ `docs/invariants/network.md` § "A split token crosses on the pipeline stream, keyed by (request, peer)"

## A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted (2026-09-27)

Substreams negotiate with multistream-select V1Lazy (`SWARMLLM_SUBSTREAM_V1=1` restores V1 for an A/B). `PeerInfo::latency_ms` and ACK samples are COSTS; a reader that means DISTANCE goes through **`network::manager::physical_rtt_ms`** (`exchange_says_lan`, `observe_network_coord`) — never compare a raw sample to a distance constant.

→ `docs/invariants/network.md` § "A substream sends with its protocol proposal; a ping sample is a COST, a distance is converted"

## A burst is paced by the link, not by a window it never fills (2026-10-20, #318)

Every QUIC connection runs quinn's BBR: **`network::transport::quic_congestion`** → the vendored libp2p-quic's ONE addition, `Config::congestion`. Sender-side only: no feature bit. A/B `SWARMLLM_QUIC_CC=cubic`; probe `examples/quic_pacing_probe`; a rig measures QUIC only with `RIG_TRANSPORT=quic`.

QUIC runs on the VENDORED `quinn-proto` 0.11.19 (`vendor/quinn-proto`, quinn PR #2976): a lossy stream is kept, not closed at "too many gaps in stream buffer" (#429). Never drop the patch on a re-vendor unless that release carries #2976; guard `quic_runs_on_the_vendored_quinn_proto_that_keeps_a_lossy_stream`. → `docs/invariants/network.md` § "A QUIC stream behind lost packets is kept, not closed"

→ `docs/invariants/network.md` § "A burst is paced by the link, not by a window it never fills"

## A speculative verify is walked where the logits are (2026-09-27)

A verify's last segment walks the drafts and answers with token ids (`LayerForward::spec_walk_at_tail`, gated at the SENDER on `features::SPEC_WALK_AT_TAIL`). `sampling::sampled_accept_reject` is the one rule; `pipeline::VerifyReply::accept` the one place a reply becomes accepted tokens; `layer_forward::spec_trailer_flags` the one writer of the flags byte. Shared noise (`coupling_seed`, `0x0B`, `features::COUPLED_SAMPLING`) is keyed by ABSOLUTE position.

→ `docs/invariants/network.md` § "A speculative verify is walked where the logits are"

**A tree that FLOWS is drawn at the tail, never walked there (#134 (b)).** `draft_tree_from` (`0x11`, after `0x10`, AAD-bound; gated at the SENDER on `features::DRAFT_TREE_FLOW` for every peer of the run; onward by `onward_refusal` + `draft_tree_from_onward`) appends rows to the segment's tree since `from`; the tail answers ONE draw per row (`sampling::draw_each_row`); the walk is the coordinator's. `model_worker::draft_tree_to_apply` is the one reading (refused unless `from ≤ index_pos` and the tree is `(index_pos − from) + positions` rows); a sampler reading history is refused (`refuse_draws_with_penalties`).

→ `docs/invariants/scheduling.md` § "A tree that flows"

## A check travels a chain like a decode step does (2026-10-02)

`PipelineExecutor::verify_may_chain` decides whether a run of remote segments gets ONE forward (every hop advertising `features::CHAINED_VERIFY`); a chained check resends on nothing (`ResendOnRefusal::Never`). **A chained run that fails is re-run UNCHAINED there** and sets the REQUIRED `chain_failed` — every caller passes `&mut self.chaining_disabled` (#318). A middle hop refuses what it cannot hand on BEFORE its worker (`layer_forward::onward_refusal`) and dials an unmet next hop (`NetworkCommand::DialPeer` via `dial_checked`, addresses through `filter_dialable`). **The sender names that target, so the dial is rationed** (`dial_next_hop`): `state.next_hop_dial_budget` (4/min per sender, 16 in all), none on a metered hold, a target the DHT did not know not looked up for 60 s (`NextHopMisses`). `SWARMLLM_CHAIN_VERIFY=0` is the control arm.

→ `docs/invariants/network.md` § "A check travels a chain like a decode step does"

## A result names the step it answers; one that did not arrive is sent again (2026-09-25)

`LayerResult::answers_step` (`0x07`) names the forward — a serving node builds it from **`LayerForward::step`**; `PendingLayerResult::expects_step` refuses any other, and every registration sets it from the forward it actually sends. Only then may a serving node resend a lost result (`resend_lost_result`, gated on `features::RESULT_STEP`). Test with `SWARMLLM_FAULT_RESULT=lose|duplicate`.

→ `docs/invariants/network.md` § "A result names the step it answers"

## A stream of verifies runs in its order, and its answers are found by number (2026-09-30)

Streamed verify forwards (`LayerForward::stream_seq`, `0x0C`, `features::STREAMED_VERIFY`) run in number order via `daemon::state::forward_streams`; an answer echoes its number (`0x08`). A stream is ONE piece of its sender's work (`dispatch::StreamWorkSlot`, `MAX_STREAM_CHUNKS_HERE`; refusals through `refuse_forward`; only to `features::STREAM_AS_ONE_WORK`). `dsd_stream::stream_shape` takes any plan whose peers' segments are ONE run (the boomerang too, #152): where this node holds the last layers the walk is ours and segments run as each answer is TAKEN, never as it arrives (#180). Several peers stream as a CHAIN (#318) only when every hop advertises `features::CHAINED_STREAM` (bit 27) and each machine appears once; a failed chain finishes in rounds, unchained (`EngineDrafter::follow`).

- **A walked check that refused a guess skips the turns built on it** (`Turn::refused_a_guess`; A/B `SWARMLLM_STREAM_TAIL_SKIP=0`), **and a wrong look-ahead too** (`LayerForward::stream_lookahead`, `0x12`, AAD-bound, `features::STREAM_LOOKAHEAD` bit 34, gated at the SENDER; A/B `SWARMLLM_STREAM_LOOKAHEAD=0`, #382).
- **A likely refusal's restart goes BEFORE the refusal** (`LayerForward::stream_branch`, `0x13`, AAD-bound, `features::STREAM_SPARES` bit 35, greedy only, #381): a `Spare` runs only on the exact refusal it names, once. Which spares go is PRICED (`dsd_stream::spare_pays`; A/B `SWARMLLM_STREAM_SPARE_LATENESS=0`, `SWARMLLM_STREAM_SPARE_MIN`). The coordinator is OFF by default (opt-in `SWARMLLM_STREAM_SPARES=1`); the far end's support is always on.
- **Every chunk is drafted with no confidence floor and is the stream's width, the last too** (`engine_drafter::stream_confidence_floor`; `draft()`'s `stop_below` is REQUIRED; `dsd_stream::chunk_guesses`): a short chunk holds the stream a round trip; a narrow tail is a new card graph late in a worker's life (#883). A/B `SWARMLLM_STREAM_DRAFT_CONFIDENCE=0.4`, `SWARMLLM_STREAM_FULL_CHUNKS=0`. → § "A stream of checks travels a chain"

→ `docs/invariants/network.md` § "A stream of verifies runs in its order, and each answer names its number"

## Every reply a serving node sends goes to whoever is WAITING — failures included

**`layer_forward::reply_target`** is the single answer to who is WAITING; it returns a `ReplyTo` nothing else can build, so `send_error_result` cannot be handed the raw sender (#707). A refusal that cannot be addressed past the hop it came from is PASSED ON by that hop: every `LayerResult` this node takes in goes through **`NetworkManager::dispatch_layer_result`**, which asks **`hop_forwards::HopForwards::take_onward_failure`** (an error from the onward peer about the step handed on, once) and reports it to the coordinator in its own name; `fail_tensor_forward` takes the forward's step as a REQUIRED argument (#305).

→ `docs/invariants/network.md` § "A chained hop's refusal goes to the coordinator"

## Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind (2026-09-26)

Every kind of peer work takes a `PeerWorkSlot`, and a refusal is ANSWERED — `layer_forward::refuse_forward` / `remote_generate::refuse_request`, worded by `peer_work_refusal()` — never silently dropped. A request that has not started leaves the last quarter of the slots alone (`admits_a_new_request`). Image-encode refusals still cannot be said (FUTURE_WORK #123).

→ `docs/invariants/network.md` § "Work a serving node will not run is refused OUT LOUD, and counted per peer whatever its kind"

## UPnP that the router refuses is said out loud, and a stopping node hands its ports back (2026-10-08)

UPnP is the DIRECT `libp2p-upnp` ≥ 0.6 dependency (it backs off a refused mapping); the facade's `upnp` feature stays OFF. A refused mapping emits no event, so `network::manager::upnp_watch::UpnpWatch` explains the silence once (`UPNP_QUIET_AFTER`); `release_upnp_mappings` closes the P2P listeners at a clean shutdown (#239). The PATCHED copy is `vendor/libp2p-upnp`. **The external port is CHOSEN, never assumed** (`vendor/libp2p-upnp/src/choice.rs`): never ask for a port another computer holds; an unANSWERED table read is `Table::Unknown` (the request fails, #950); never an `expect` on the mapping table (libp2p #6377); a missing gateway is searched again (2 → 30 min, #329). A relay-only computer is told what is in the way (`UpnpWatch::verdict`, `UpnpVerdict`, via `SharedState::reachability_json`).

→ `docs/invariants/network.md` § "UPnP that the router refuses is said out loud"

## Every node keeps a home relay, reachable or not (2026-11-16, #464)

Every node reserves once it reaches a bootstrap relay (`relay::HOME_RELAY`) and the liveness tick re-arms after a lost circuit, whatever its reachability. `relay::reason_to_reserve` is the one decision (a reachable anchor keeps none); `relay::bootstrap_relay` the one reading of a bootstrap entry as a relay (never this node). **The latch follows the relay LISTENER** (`relay_listener`). **A refusal backs off** (`relay::relay_retry_gap`, 20 s doubling to 10 min, reset by a granted reservation; only the relay's ANSWER counts, `relay::relay_refused_us`) **and the next try asks another relay** (`relay::relay_preference`: connected first, the refuser last, the rest by a hash of our id — never config order, #467). Never in offline mode; never depend on AutoNAT's "unreachable". "Direct" stays `publicly_reachable`, which counts no circuit.

→ `docs/invariants/network.md` § "Every node keeps a home relay, reachable or not"

## A relayed circuit lasts while both ends hold it (2026-11-17, #473 (c))

Our relay server sets no byte limit (`max_circuit_bytes` 0) and only a 30-day backstop duration: the ends redial at once, so a limit only cuts the pair. A node that stops relaying ends the circuits it carries (`relay::circuits_to_end`, the discovery tick). Never add a total deadline a redial undoes.

→ `docs/invariants/network.md` § "A relayed circuit lasts while both ends hold it"

## A hole punch offers our confirmed addresses, handed straight to DCUtR (2026-11-16, #466)

DCUtR's `CONNECT` carries only address CANDIDATES, and libp2p-swarm drops a candidate that is already confirmed — so a
UPnP-mapped or declared address never reached a hole-punch partner, and a peer that met a reachable node by relay
stayed there. `offer_our_addresses_for_hole_punch` hands each confirmed direct internet address
(`relay::offer_for_hole_punch`) to DCUtR's own `on_swarm_event`, on confirmation and every liveness tick. **Never raise
it as a swarm candidate** — dropped silently (gotcha #1122). → `docs/invariants/network.md` § "A hole punch offers our
confirmed addresses, handed straight to DCUtR"

## A lost peer is looked up; peer exchange names every peer in turn (2026-11-16, #465)

A peer that named a direct internet address at any Identify this run (`lost_peers::worth_looking_up`, remembered by
`note_dialable` apart from the registry) is noted lost on its last close, whatever the branch, and looked up in the DHT — `get_closest_peers`, whose walk dials it — a minute after the loss, then at a gap doubling
to an hour, for a day (`network::manager::lost_peers::LostPeers`, four a liveness tick); both DHT lookups read
their answer through `addresses_the_dht_found` (public only). A peer-exchange answer is built ONLY by
**`helpers::pex_answer`**: two addresses a peer (its best, then its best over the other transport), connected peers first, a new random order each answer — never the
registry's own order, which named the same few peers in every answer. → `docs/invariants/network.md` § "A lost
peer is looked up, and peer exchange names every peer in turn"

## Offline mode dials nothing on the open internet — one gate, at the transport (2026-11-17, #468)

`network::transport::OfflineGate` refuses, while `credits.offline_mode` is on (read at each dial), every dial of an address `pool::invite::multiaddr_is_internet_reachable` calls the internet (a public IP or any DNS name) before a byte is sent, whichever behaviour asked. It sits OUTSIDE the resolver and INSIDE the relay client, in `network::transport::swarm_transport`, the one builder of the TCP + QUIC + DNS stack: never add a transport around it in the manager (`.with_tcp(`, `.with_quic()`, `.with_dns()` there fail `every_transport_the_node_builds_is_metered_by_protocol`), never gate per dial site. Refused with `OfflineRefusal`, never `MultiaddrNotSupported`. An ARRIVING connection from such an address is dropped before noise; one open before the switch is closed by the liveness tick (`close_internet_connections_while_offline`). A VPN or overlay range (Tailscale's 100.64/10) counts as this computer's own network — never describe offline mode as "no packet crosses the internet". Prior art: go-libp2p's `ConnectionGater.InterceptAddrDial` / `InterceptAccept`.

→ `docs/invariants/network.md` § "Offline mode dials nothing on the open internet"

## "Is this connection direct?" — `network::relay::addr_is_direct_transport`

**`network::relay::addr_is_direct_transport`** is the single answer, for both layers that choose a connection: a real `/ip4`, `/ip6` or `/dns*` hop AND no circuit. `is_relay_circuit_addr` alone misses the listener's view; the vendored request-response inbound handler records the send-back address. `peer_direct_conns` is gated on it; any new "prefer direct" logic uses it (gotchas #356, #179). Keep the loop-stall tripwire in `NetworkManager::run`.

→ `docs/invariants/network.md` § ""Is this connection direct?" — `network::relay::addr_is_direct_transport`"

## A relayed body is its origin's only once its origin is PROVEN

An ephemeral seal is anonymous (`crypto::relay_seal`): `origin` in the AAD is a claim anyone can make. A relayed message or tensor is dispatched as `Some(origin)` only when `relay_seal::Opened::Proven` (the origin's signature inside the seal), `RelayReplayGuard::admit` takes it (refuse when full, never forget early), and the inner is `is_relay_eligible`; nothing is learned from a body before that (`take_proven_relayed_origin`). A node relays ONLY to a target that `target_speaks(RELAY_SIGNED)`; refusals go through `refuse_relayed`; a new relayed kind gets its own `RelayedKind`, signed (#298).

**A share is counted by NETWORK, never by key or a name the peer chose (#299):** `network::helpers::NetGroup` (public IPv4 /16, IPv6 /32, a private host, a host name only if `ConfiguredNames` has it, else `Unaddressed`; a circuit is `Circuit` of OUR latest DIRECT connection to its relay — `manager::relay_network`, `PeerRemote::direct`, `record_peer_remote` — never the relay's own share), applied in `admit_relayed`, the one admission. A proven origin's registry entry goes through `identify::evict_one_registered_peer` / `unconnected_registry_entries` (#403; `MAX_UNCONNECTED_REGISTERED`) and its mapping through `remember_peer_id` — never a bare insert.

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

A dial that fails `WrongPeerId` proves the (peer, address) pair wrong: `NetworkManager::forget_address_answered_by_another` (from `dht::pair_proven_wrong`) removes exactly that pair from Kademlia, the peer registry and the peer cache (`stale_peers::ProvenWrong`, #413). Never back off a PEER on it. A peer evicts only on its OWN unreachability: out of reach for `stale_peers::UNREACHABLE_FOR` (across a restart too: `LastReached::seeded`, #472), its next failed dial (a transport error or `WrongPeerId`, never `Aborted`/`Denied`) takes it out of Kademlia (`leave_the_table_if_out_of_reach`) and `dial_bootstrap_peers` stops seeding it; Identify puts a connected peer back (#413).

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

`SwarmMessage::PromptAdmissionAsk` / `PromptAdmissionAnswer`, gated at the SENDER on `features::PROMPT_ADMISSION` (bit 25). The coordinator's wait is keyed `(ask_id, NodeId)` — a fresh id per QUESTION (`pending_prompt_admissions`) — and dispatch resolves only the AUTHENTICATED sender's key (`resolve_connected_peer_id_bytes`). The asked node drops a malformed ask (`ask_is_well_formed`), answers a metered hold with `Refused`, rations weighing (`AskBudget`: 12/min per peer, 60/min in all) and weighs under a FRESH correlation id (`weigh_peer_prompt`): anything keyed or held by a request id is a peer's lever on a live request; the question may NAME it only to leave its own caches out (`IpcAdmitPrompt::weighs_for`, #448). A refusal is ONE class (`ServiceUnavailable`) repeated through `peer_words`; a missing answer is "go ahead". `Refused::counted` (`#[serde(default)]`) says what `layers_that_fit` counts; a NEW `LayersCounted` variant needs a feature bit. Worker side: `arch-worker-memory.md`.

→ `docs/invariants/memory.md` § "A split's peers are asked about room before the pass too"

## A request someone waits on is sent until answered (2026-10-15)

A gossip publish that succeeded says nothing about who received it. A pool join is resent (`pool::manager::JoinResend`) until accepted or its window ends — said out loud; the owner answers a repeat from the code's claimer with the SAME invitation (`PoolInviteCode::claimed_by`), at most once per `REPEAT_JOIN_ANSWER_GAP`. The CLI reports the outcome, not the send.

→ `docs/invariants/network.md` § "A request someone is waiting on is sent until it is answered"

## A metered connection holds back what this node uploads for others (2026-10-15, #277)

`SharedState::metered_hold` answers for uploads too; the network and dispatch loops and the capability read `SharedState::metered_hold_now`, never the system. While it holds: the capability withdraws inference (`can_serve_inference`), a new request is refused out loud through `dispatch::admits_a_new_request` (its metered answer REQUIRED; `ServiceUnavailable`, re-plan without penalty), parts and prompt-cache snapshots are answered as a miss; running work, the owner's requests and gossip go on. No LAN exemption (`is_lan_peer` cannot tell a VPN peer).

**Relaying**: `SharedState::relay_forwarding_enabled` is the one answer (read live; the hold stops the AUTOMATIC donation only). It reaches the server — `network::relay::OnlyWhileRelaying`, a required `build_behaviour` argument — and its DHT record is withdrawn on the falling edge (`discovery::stop_providing_relay_service`). Gotcha #861.

→ `docs/invariants/network.md` § "A metered connection holds back what this node uploads for others"

## A hot computer takes no new work for others (2026-11-11, #440)

`SharedState::peer_work_hold_now` is the one answer to "does this node take a peer's NEW request": `PeerWorkHold::Metered`, `Updating` (#444: `update_restart::drain` → `arch-state-and-config.md`) or `Heat` (`daemon::host_heat`: the card hot or cooling for `CARD_HOT_FOR` (120 s), its hardware protection on, the processor ≥ 93 °C for 5 min, or `nvidia-smi` stuck a minute; resumes after `COOL_FOR` (60 s)). **Unknown is never hot**: a sensor silent `READING_GONE_AFTER` (5 min) is forgotten; sensors are read off-loop within `SENSOR_READ_LIMIT` (40 s) (`refresh_host_heat`). Every dispatch admission takes it as a REQUIRED argument (`admits_a_new_request`, `admit_own_forward`, `admit_stream_chunk`); prompt admission answers with it (`hold_refusal`); `can_serve_inference` reads the same predicate. Refused `ServiceUnavailable` (`heat_refusal`): re-plan, no penalty. **Running work stops too** (`PeerWorkHold::Heat { stops_running }`, latched): after `RUNNING_WORK_STOPS_AFTER` (120 s), at once on a stuck driver or hardware protection — forwards refused (`CoolingStopsRunning`), replies aborted (`inbound_generate_aborts`) so the coordinator fails over (#236). Off: `SWARMLLM_HEAT_GUARD=0` — every script that starts TEST nodes sets it; never the safety kit (`live_node_swap.sh`). Fault: `SWARMLLM_FAULT_HOST_HEAT=card|protecting|processor|driver|cool`, or `file:<path>`.

→ `docs/invariants/network.md` § "A hot computer takes no new work for others"

## A computer that shares nothing may look at the swarm, not use it (2026-11-04)

`SharedState::swarm_use_of_peer` (the dispatcher, for every request a peer STARTS here: `admits_a_new_request`, a tensor-parallel prompt pass, every stream prompt piece `admit_stream_chunk`) and `swarm_use_here` (`pool::scope::request_scope`; the scheduler's `TooFewPartsToUseTheSwarm` / `TooFarBehindToUseTheSwarm`) are the one rule, `swarm_use::verdict`: at least `PARTS_TO_USE_THE_SWARM` (2) parts announced AND fewer than `RELEASES_BEHIND_TO_LOSE_USE` (2) releases behind the newest THIS node's update check found SIGNED (`UpdateInfo::signature_verified`) — never a peer's claim (`PeerVersionWatch`'s rule).

- A forward is judged as the coordinator it names, never the hop that chained it (`layer_forward::judged_as`, `swarm_use_of_forward`; `features::JUDGES_THE_COORDINATOR`, bit 38).
- Judged from what the peer says of itself: its capability's version and the MORE of its capability's and announced parts; this node by `own_shard_announce`.
- A capability naming an older version than held is never installed (`swarm_use::capability_may_replace`, the one installer), nor on GOSSIP one of the SAME version not built later (`NodeCapability::issued_at_ms`, `#[serde(default)]`; `capability_news` carries it over); one sent DIRECT is taken whatever its stamp; a refused one's latency readings are not merged (#402).
- No capability yet → allowed only while CONNECTED, else `NotHeardFrom` (#403); one gossiped BEFORE its entry existed is held (`swarm_use::HeardBeforeKnown`) — never for a CONNECTED computer (#405). A pool device → allowed; no LAN exemption.
- The refusal is this node's OWNER's: never raised for a split led here for a peer; `ServiceUnavailable`-worded (`SwarmUse::peer_refusal`): bar and re-plan, no penalty. Behind is counted in PUBLISHED releases (`ReleasesKnown::behind`). The PARTS half withholds only use; the VERSION half costs connections (next rule). Display: `swarm_use_json`, `NodeCapability::{auto_manage, updates_itself}`.

→ `docs/invariants/network.md` § "A computer that shares nothing may look at the swarm, not use it", § "A computer this node has not heard from is not judged allowed"

## A computer two releases behind loses contact, not only use (2026-11-04)

`SharedState::cut_off_for` is the one judgment (never a pool device, never a computer this node bootstraps from — `SharedState::bootstraps_from`, Bitcoin Core's `noban` — nobody while the pool state is busy); `NetworkManager::cut_off_peer` the one actor, on the network loop: at Identify, before registering or catching up, on the release the agent version names (`swarm_use::agent_version`, `swarmllm/<version>` from v0.3.271; libp2p's own `rust-libp2p/…` reads as v0.3.270 at most — `swarm_use::release_in_agent`, #415, #1079), and on the liveness tick over `peers_own_versions` — only what the computer stated ITSELF (`note_capability_heard`), never a gossiped or DHT one. The cut blocks it (`allow_block_list`), drops it from Kademlia, and `dial_checked` / `try_enqueue_redial` skip it, for `CUT_OFF_FOR` (10 min), at most `MAX_CUT_OFF` (1024) at once. A relayed body whose PROVEN origin is cut off is refused (`origin_is_cut_off`). One release behind keeps everything; a cut computer still updates from GitHub.

→ `docs/invariants/network.md` § "A computer two releases behind loses contact, not only use"

## A peer's catch-up that lands before its Identify is held, not dropped (2026-11-06, #415)

**`NetworkManager::dispatch_authenticated`** (the one choke point) HOLDS a catch-up kind from an unmapped peer — `NodeCapabilityUpdate` within `dispatch::capability_fits`, `ShardAnnounce` within `fits_one_message`, `NicknameGossip` within `validate_nickname`; 8 messages and 1 MiB a peer, 8 MiB in all, 32 peers and 4 a `NetGroup`, 10 s (`network::manager::before_identify`) — and the Identify handler hands them on in order once the peer is mapped, registered and connected; a refused or gone peer has them dropped. A new kind a peer sends as it identifies us joins `is_a_catch_up`; never hold a forward or anything unbounded.

→ `docs/invariants/network.md` § "A peer's catch-up that lands before its Identify is held, not dropped"

## Every part this node serves shares ONE upload limit (2026-10-24, #335)

`network::manager::upload_pacer::UploadPacer` is the one timeline: an answer books its share of `shard_upload_mbps` (read live) BEFORE the read, for the bytes the part HAS past the offset (`bytes_to_book`), and is sent when its share ends — never a sleep per request. An answer holds at most 4 s of the limit and is released within `RELEASED_WITHIN` (two thirds of `SHARD_STALL_SECS`). Past that, or with the inbound queue full, `answer_part_request_busy` answers BUSY to a `features::SHARD_BUSY` downloader (bit 32; it resumes at its offset) and empty to an older one. Bound every wait a shared limit adds by the requester's patience. **No network holds more of the queue than a node at the default concurrency books** (`NETWORK_HOLDS_AT_MOST` = `DEFAULT_MAX_CONCURRENT_DOWNLOADS` × `ANSWER_HOLDS_AT_MOST`, 12 of 20 s, #340): booked per requester network (`PeerRemote::group`, `NetGroup`), busy past it (`Grant::Busy::network_share`). A/B `SWARMLLM_UPLOAD_SHARE=0`; rig `split_rig.sh fetch` with `FETCH_OTHER`.

→ `docs/invariants/network.md` § "Every part this node serves shares ONE upload limit"

## Centralised Wire-Format Helpers

Single sources of truth for invariants that silently break at the wire if duplicated. Read the evidence before changing one.

- **`network::protocol::build_layer_forward_aad`** — ⚠ a new trailer is NOT a no-op for an older peer: feature-gate every optional trailer at the SENDER and bind it in this helper (encrypt and `decode_layer_forward_encrypted` both use it); clamp a clamped trailer AFTER the AAD is rebuilt.
- **`network::pipeline_stream::chunk_layer_forward`** — the one splitter for chunked sends.
- **`SharedState::resolve_pending_layer_result`** — the ONLY way to deliver a `LayerResult` into `pending_layer_results`.
- **`daemon::dispatch::timestamp_fresh_one_sided`** — the staleness primitive (`gossip_timestamp_fresh` wraps it); **`credit::ledger::check_signed_freshness`** for signed `DateTime<Utc>` messages.
- **`pipeline::pack_verify_tokens_to_le_bytes`**, **`pipeline::build_spec_verify_forward`**, **`pipeline::build_kv_truncate_forward`**, **`pipeline::register_pending_layer_result`** — shared builders for verify/truncate forwards and waiter registration.
- **`storage::Database::with_write_table`** — the write-transaction wrapper; hands the worker off before it blocks (#314).
- **`swarmllm_types::ShardResponse::empty()`**, **`ShardResponse::busy(retry_after_ms)`** (only to a `SHARD_BUSY` downloader), **`swarmllm_types::LayerResult::error(request_id, reason)`** — canonical refusal/error replies.
- **`network/manager/connections::try_enqueue_redial`** — dedup + cap + push for `pending_redial`.
- **`responses::types::raw_tool_kind_or_unknown`**, **`cli::bail_if_no_api_key` / `cli::exit_daemon_unreachable`** — shared error wording.
- **`model::auto_manage::spawn_check_and_load`** — "shard landed → reload model → refresh dashboard", always the three steps together.
- **`pool::invite::{encode_invite_code, decode_invite_code}`** — the `swarmpool://` v2 codec (`looks_like_v2` routes it from the legacy path); **`pool::invite::addresses_for_invite`** is what a code carries: the outbound route's own address (`outbound_addresses`) + internet-reachable ones, never a Docker bridge or an overlay off the default route; a CGNAT route (a Tailscale exit node) keeps the private LAN too.

→ `docs/invariants/network.md` § "Centralised Wire-Format Helpers"

## The models peers name are bounded, and one in good standing is never displaced (2026-10-24, #333)

`SharedState::admit_a_peers_manifest` (`daemon::state::peer_models`), asked LAST in `judge_peer_manifest` (`PeerManifest::NoRoom`), is the one bound: 1,024 models adopted as new from peers, 64 MiB of their stored JSON (`weight_of`), 1 MiB a model. Room comes only from a model NOBODY holds — this node or a CONNECTED node (`SharedState::holder_standing`; never `peer_registry`) — that this node has not acted on (`acted_on_model`), oldest first, 64 looks per ARRIVAL (CLOCK); none to spare refuses the newcomer. The wishlist is no protection. A forgotten model's stored row goes too; `register_manifest` stores a peer's manifest only for a model with a part here. The door refuses an id past 256 bytes, a part number ≥ 256, a name past 512 bytes. Holder records: 4,096 per KEY (`MOST_PARTS_A_HOLDER`).

→ `docs/invariants/network.md` § "The models peers name are bounded"

## A full holder list takes a stranger's claim only in a stale record's place (2026-10-25, #339)

A node not connected here is displaced first and aged out, never refused. `SharedState::holder_standing` is where a holder stands (not connected, or connected from a network: `connected_networks`, `NetGroup`) for the registry's hook AND #333's `holder_counts`; asked with NO registry guard held. `holder_to_give_up` at `MAX_HOLDERS_PER_SHARD`: a stranger takes only a stale (`STALE_HOLDER_AFTER`, 2.5 h) STRANGER's place, else is ignored; a connected node whose network already holds the most places takes its own network's NEWEST arrival's place. `age_out_holders` (every `FULL_REANNOUNCE_EVERY_TICKS` ticks) forgets unconnected records past the bound. **A DHT provider record is the DHT's word, never the holder's** (#341): `merge_dht_providers` writes `HolderRecord::first_hand = false` through the one writer `record_holder` (not counted by `part_has_a_holder_where`, aged out whoever it names).

→ `docs/invariants/network.md` § "A full holder list takes a stranger's claim only in a stale record's place"

## Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer

Publish on CHANGE, never per item per tick — gate on a digest of what is ASSERTED, excluding any timestamp. Gossip only what this node MEASURED (`local_region_demand`) and of that only work served FOR PEERS (`SharedState::note_request_served_for_a_peer`; `announced_score`, #389 (a); `models_served` by the DAY via `auto_manage::unused::models_served`, #452) and only what a reader uses (#91); size the mesh from the PEERS the cache names (`discovery::peers_named`), not addresses (#91 (1b)).

- **Catch a newcomer up POINT TO POINT** (`NetworkCommand::SendDirectMessage`): capability AND parts list at identify time (`ModelMgmt::own_shard_announce`, #359) and our name while shown (`identity::nickname::own_record_for_a_newcomer`, #417).
- **Don't repeat what the swarm just HEARD** (`state.models.manifest_heard`, `MANIFEST_QUIET_WINDOW`, GOSSIPED arrivals by VERSION); a version heard before this node last changed its OWN is not heard (`manifest_needs_broadcast`, #224).
- **Unchanged capability and parts list repeat at a doubling wait** — 5, 10, 20, 40 min (`health::monitor::QuietRepeat`, Trickle; no gap passes a third of the 50 min `STALE_HOLDER_AFTER`; A/B `SWARMLLM_QUIET_REPEAT=fixed`, #91).
- **A model whose last part went is NAMED, never left out**: `shard_announce_round` keeps it in `complete_for_models` with no parts for `RETRACTION_REPEATED_FOR` (`retracted_models`) — a connected holder's claim never ages out, so silence kept it credited on every peer (2026-11-17). → § "A model this node stopped holding is retracted by name"
- **`health::monitor::capability_news`** compares whole capabilities, carrying over only uptime, the latency hint, free RAM where its one reader (`memory_for_model_layers_mb`) would not see it, and figures inside their deadband (memory 256 MB, disk 1 GB, coordinate 5 ms, decode timing 1.5x); a new field is news by default. Liveness is `HealthPing`'s job.
- **`features::MANIFEST_DIGESTS` peers get VERSIONS (`ManifestDigest`) and pull (`ManifestPull`) what they lack**, at catch-up and every `MANIFEST_QUIET_WINDOW` (`health::monitor::send_manifest_digest`; `daemon::state::manifest_sync`, #91 (2)). A new connection is a new catch-up (BEP 3): `ManifestSync::note_new_connection`, told again when its capability arrives (`note_capability_heard` → `want_catch_up` → `catch_up_new_connections`; `ANSWER_AGAIN_AFTER`). The gossiped repeat waits `MANIFEST_QUIET_WINDOW_WITH_DIGESTS` (24 h) only while EVERY connected peer takes digests; neither digest message goes on gossip.

→ `docs/invariants/network.md` § "Gossip says what CHANGED, to everyone — and what one peer lacks, to that peer"

## A parts list longer than a receiver takes goes in pieces, each model whole (2026-11-02)

Every node drops a `ShardAnnounce` past **`swarmllm_types::MAX_SHARDS_PER_ANNOUNCE`** (512) unread, so a sender never sends one: **`network::manager::commands::pieces_if_too_long`** (asked by the gossip publish and the direct send, never at call sites) cuts it with **`ShardAnnounce::into_pieces`**, each model's parts, tags and flags in ONE piece. No wire change; never raise the bound. **Every list in it is bounded by ONE predicate on both sides** — `ShardAnnounce::fits_one_message`: parts, `origin_checked_models` AND `complete_for_models`; a new list joins it.

→ `docs/invariants/network.md` § "A parts list longer than a receiver takes goes in pieces"

## A peer's clock may run minutes fast; gossip's gate allows 5 (2026-11-06, #416)

`network::manager::events::gossip_out_of_time` gates timed gossip (health pings, parts lists, credit gossip): at most `GOSSIP_MAX_AGE_SECS` (300) old and `GOSSIP_AHEAD_SECS` (300) ahead — credit gossip the ledger's `CLOCK_SKEW_TOLERANCE_SECS` (30) — one-sided (gotcha #44). A kind joins the gate only if kept per sender; the other timed checks are weighed one by one (FUTURE_WORK #416) — never one shared constant raised for all. The owner is told at 20 s off GitHub's `Date` (+ `Age`) header (`update::note_clock`, `CLOCK_OFF_TELL_SECS`), never by peers' stamps. A nickname's stamp is checked ONLY by `NicknameRecordExt::verify` (5 min ahead, 24 h old); a new record is stamped after the previous one (`new_signed`'s REQUIRED `after`).

→ `docs/invariants/network.md` § "A peer's clock may run minutes fast"

## A peer that reconnects beside an old connection is told our subscriptions again (2026-11-05)

Gossip runs on the VENDORED `libp2p-gossipsub` 0.49.5 (`vendor/libp2p-gossipsub`, `[patch.crates-io]`): **`Behaviour::send_our_subscriptions`** re-sends every subscribed topic when a connection opens beside another (`other_established > 0`) AND when one closes beside a live one. Never drop the patch on a re-vendor; guard `gossip_runs_on_the_vendored_copy_that_tells_a_reconnected_peer_our_subscriptions`; the crate's own test runs in CI `vendored.yml`. Symptom: a peer whose `uptime_seconds` does not advance between two reads. **A node closes every connection before it restarts into an update or stops** (`NetworkManager::close_every_connection`, #407).

→ `docs/invariants/network.md` § "A peer that reconnects beside an old connection is told our subscriptions again"

## A computer's country is read as two letters, in one place (2026-11-05, #407)

**`api::admin::country_code`** reads every country — a peer's text and this node's own lookup: two ASCII letters, never `XX`. **A peer's region is read as it ARRIVES**: the capability installer goes through **`api::admin::capability_as_kept`**; a new installer calls it. This node's own is `effective_region` / `_sync`: the LIVE `identity.region` through `country_code`, else the lookup (`background::region_retry_after`). **`api::admin::map_region_of`** places a peer on BOTH map paths (`network_map`, WebSocket `region_summary`): its country, ours only for a LAN peer, else `??` — counted, drawn nowhere.

→ `docs/invariants/network.md` § "A peer that reconnects beside an old connection is told our subscriptions again"

## A counter named for an outcome may only be counting the attempt

GossipSub's `sent` / `sent_bytes` count ATTEMPTS per recipient, so `sum(topic.sent_bytes) <= out_bytes` must NOT be asserted — the gap is the signal. Read both drop paths (queue expiry via metrics; queue full only via `gossipsub::Event::SlowPeer`). `sent_msgs` is not a publish rate; `published_msgs` is. Before estimating a total's components, look for the counter you are not reading (gotcha #673).

→ `docs/invariants/network.md` § "A counter named for an outcome may only be counting the attempt"

## A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT

The per-shard tensor table is ~92% of a manifest and is derived, not sent: **`daemon::shard_loader::derive_tensor_entries`** rebuilds it as a FALLBACK. Every table — published or derived — is written by **`LayerShardLayout::tensor_entries`** (#334 (2)). ⚠ `manifest.shard_count` is the layout's OUTPUT, so the count is SEARCHED for with the published `size_bytes` as oracle; return `None` rather than a partial table; do NOT "fix" the underproduction. `a_derived_tensor_table_matches_the_published_one` (ignored; `SWARMLLM_TEST_MODEL_DIR`).

→ `docs/invariants/network.md` § "A manifest's tensor table is DERIVED data, and the shard count is its OUTPUT"

## A request reaches only the nodes its scope names, on every path (2026-10-12)

`pool::scope::request_scope(shared, model_id, asked)` is the single answer to which nodes a request's data may reach — "Keep on this computer" by the owner (per model) or by the REQUEST (`KeepAsked`, #251 (2b): `api::keep_here` reads the header, the body's `swarmllm` field or MCP's `_meta`; strictest wins, unreadable is a 400) → this node alone, else private mode's set; `allowed_node_set` is for shard management only. `InferenceRequest::local` takes the keep as a REQUIRED argument; a path that writes `KeepAsked::Unasked` is listed in `KEEP_UNASKED_IS_THE_ANSWER`. A kept request's prompt-cache entries are never announced or exported (`prefix_cache::Entry::kept_here`); cloud paths ask `providers::cloud_refused_for`. A failure is never broadcast. **Another pool's listing is acted on only through `pool::scope::listed_pool_members`** (`share_model_catalog_min_members`); a pool lists a model only from its devices' OWN holder records (#443).

→ `docs/invariants/network.md` § "A request reaches only the nodes its scope names, on every path"

## Network coordinates: publish them rough, feed them a MINIMUM

`NodeCapability.coord` (Vivaldi, `features::NETWORK_COORDS`) is published however rough (`is_usable()` is the CONSUMER's gate). Feed the windowed MINIMUM, never a raw sample — `SharedState::observe_network_coord` is the single writer, `LatencyFilter` takes the minimum (not Serf's median); size a window against its fill rate (`LATENCY_WINDOW_MIN_SAMPLES`, `LATENCY_SAMPLE_MAX_AGE_MS`).

**Two coordinates (#91): the capability's MOVES only when the node has moved; the timed answer is the one as it stands.** `netcoord::UpdateFilter` (Ledlie et al.'s RELATIVE heuristic; change past 0.3 of the round trip to the nearest neighbour, floor `UPDATE_MIN_MOVE_MS`) is fed only by `observe_network_coord` and read only by `network_coord_for_publication`; a peer-exchange answer carries `network_coord_now` (`PeerExchangeResponse::coord`), preferred by the asker (REQUIRED `sent_with_it`). Never a fixed threshold on the raw coordinate; a restored coordinate is NOT shipped. A/B `SWARMLLM_COORD_FILTER=0`; DIAG `published network coordinate moved`.

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
