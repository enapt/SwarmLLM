# Changelog

All notable changes to SwarmLLM are documented here.

## [0.3.12-alpha] — 2026-07-23

### Fixed

- **Testing a Moonshot (Kimi) or DeepSeek API key no longer wrongly fails.** When
  you pasted one of those keys and pressed "Test", the check tried a model name
  the provider had retired, so a perfectly good key came back looking invalid.
  It now tests against the current model, and the Moonshot "get a key" link
  points at the international site instead of the China-only one.
- **The prebuilt Linux download now runs on more systems.** People on Debian 12
  and other systems with an older core library had to compile SwarmLLM
  themselves, because the ready-made Linux build was made on a newer system than
  theirs and refused to start. It's now built against an older baseline, so it
  runs on Debian 12, Ubuntu 22.04 and later, and most current Linux systems with
  nothing to compile.
- **A computer penalised into a deep credit deficit now recovers on its own.**
  Credits set your priority and how much you can lean on other machines; they
  never stop your own use. A machine driven far below zero — by heavy borrowing,
  or by an earlier accounting bug — used to stay there indefinitely, stuck at the
  lowest priority and unable to share the pool's work. Its balance now drifts
  back toward zero over time, only ever upward and only while it is negative, so a
  one-off penalty is no longer permanent. Machines in good standing are
  unaffected.
- **A model you asked for could sit unusable behind one stuck download.** Models
  are fetched a piece at a time. When a piece stalled, the software immediately
  tried that exact same piece again, over and over — and while it was stuck, a
  different model you had just asked for couldn't start downloading at all, so it
  stayed incomplete and unusable no matter how long you waited. A piece that
  fails to download now waits before being retried, and waits longer each time it
  keeps failing, so one stuck download no longer holds up everything else. The
  model you asked for starts downloading right away.
- **Activity messages now show a device's name, not a code.** Several messages in
  the activity log (a device leaving, being removed from a pool, or sharing a
  model piece) showed a raw identifier even when you'd given the device a
  nickname. They now use the nickname wherever one is set.
- **The "local storage is full" notice now actually appears.** If your browser's
  local storage filled up while saving chat history, the warning meant to tell
  you was quietly dropped. It now shows.
- **One automated check kept rebuilding from scratch.** A step in the project's
  own build pipeline stored its cache under a name GitHub rejects (it contained a
  comma), so the cache was never reused and that check rebuilt everything every
  run. It now uses a valid name and reuses its cache. This affects only the
  project's automated checks, not anything you run.

## [0.3.11-alpha] — 2026-07-22

### Fixed

- **Computers on the public internet kept trying to reach home networks.** A
  machine remembers where it has seen others, and those addresses include the
  ones only reachable inside someone's house — their home network, and virtual
  network adapters created by other software. A server on the internet can
  never reach any of those, but retried all of them every minute regardless. It
  now only keeps trying addresses it could actually reach from where it is.

  Machines on a home network are unaffected and still remember each other's
  local addresses, which is how two computers in one house find each other
  again after a restart. A laptop that moves between networks keeps both sets,
  so nothing is lost by being on the wrong one when it last saved.

## [0.3.10-alpha] — 2026-07-22

### Fixed

- **The dashboard showed no readable text.** Every label, button and heading
  that comes from a translation file appeared as its internal name instead of
  words. A fault in the test-model panel added in 0.3.9 ran during startup and
  stopped everything after it, including the step that applies translations.
  That panel can no longer take the rest of the interface down with it.
  **Anyone on 0.3.9 should update.**
- **"No models available" now explains itself.** It used to say that and
  nothing more, which tells you something is wrong without saying what or what
  to do. It now says whether it is still looking for other computers, has found
  none, or has found some that are not sharing anything yet — and offers a way
  forward in each case rather than leaving you at a dead end.
- **A model said "Installed" when it was not.** The test-model list treated
  simply having heard of a model as having it, and a machine hears about a
  model as soon as any other machine mentions it. It now means what it says:
  this computer is actually storing part of it.
- **A backup copy of a model folder was offered to the network as a real
  model.** Copying a model folder — to `.FULLBACKUP`, `.old` or similar — made
  a model that other machines recorded, counted, and could never obtain,
  because its name came from the folder rather than the model. Such folders are
  now ignored, with a note in the log saying how to fix it if the model is
  genuine.

### Added

- **More in Copy diagnostics.** It now also reports the addresses this computer
  can be reached at, and how many remembered addresses are actually usable — the
  two things most often needed when working out why a machine cannot be found.

## [0.3.9-alpha] — 2026-07-22

### Added

- **A shared test model you can get in one click.** A new machine with no
  models, on a swarm that has nothing to offer yet, previously had nowhere to
  go — the chat screen could only list what was already out there. It now
  offers a small shared model instead, so there is something to actually do.

  The same model answers a request from people testing across several
  machines: if everyone runs the same one, speed results can be compared.
  Otherwise each person measures a slightly different version and the numbers
  were never alike to begin with.

  Three sizes are offered — a tiny one that just checks the network works, a
  recommended one that runs on almost any computer, and a large one for
  testing the limits of powerful machines. You can pick one from Settings, from
  the chat screen when nothing else is available, or tick a box during first-run
  setup. That box is off by default: it is a real download, and most people
  setting up want something to chat with rather than a test model. Test models
  are labelled wherever they appear so they are not mistaken for a
  recommendation.

  By default you get only the share your machine should hold rather than the
  whole thing, since the point is for the swarm to serve it together. Getting
  the whole model is offered separately.
- **Copy diagnostics.** A button in Settings that copies a short summary of
  your node — version, connections, models, recent events — ready to paste when
  reporting a problem. It contains no keys, invite codes or file paths.

### Fixed

- **Release builds were slow again.** The container image build was filling
  GitHub's shared build-cache allowance and pushing out the cache the Linux GPU
  build depends on, which took that build from about 10 minutes back to nearly
  an hour. Image layers are now cached elsewhere, and the two image builds no
  longer overwrite each other's cache.

_(0.3.8-alpha was tagged and withdrawn before publishing — a mistake in the
build-cache change above stopped the container images from building. Nothing
was released under that version.)_

## [0.3.7-alpha] — 2026-07-22

### Fixed

- **A machine could keep being sent work it was no longer able to do.** When a
  computer stopped hosting part of a model — you deleted it, storage cleanup
  removed it, or automatic management moved it elsewhere — other machines were
  never told. They carried on treating it as a place to send that part of the
  work, and requests routed there timed out or failed with "no standby
  available". Waiting did not help, because there was no mechanism by which the
  news could ever arrive. Machines now say which models they are giving you the
  complete picture for, and anything left out is dropped. This also repairs
  "remove model", which has always tried to announce that it no longer hosts
  anything and was never heard.

  Both machines need this version for it to work between them. Mixed versions
  keep the old behaviour rather than breaking, so there is no need to upgrade
  everything at once.
- **Requests were charged for tokens they never used.** A request sets aside
  credits up front based on the *most* it could generate, and that whole amount
  was kept even if the answer came back in one word. At the default settings
  that is 20,480 credits for every request regardless of length — one operator
  reported a balance of -41,400 after a handful of attempts. Requests are now
  charged what they actually used, and the difference is returned. This matches
  how requests below the reservation threshold were always charged, so the two
  no longer disagree.

## [0.3.6-alpha] — 2026-07-22

### Fixed

- **A GPU node could auto-update itself into a CPU-only build.** The updater
  matched a release asset on operating system and processor only, and the only
  bare binaries published were the CPU ones — so a machine running the CUDA or
  Windows-GPU build downloaded the CPU binary and installed it over itself,
  silently losing GPU acceleration. Every variant now publishes its own binary,
  and the updater will only install one built the same way as the one running.
  If nothing matches, it reports no update rather than installing something
  else. Auto-update is off by default, so only nodes that opted in were
  affected.

  Upgrading a GPU node that already has auto-update enabled: one more update
  may still fetch the CPU build, because the *currently installed* binary is
  the one choosing. Reinstall the GPU archive once and it will track the right
  variant from then on.
- **Nodes kept re-dialling addresses that could never work.** Remembered peer
  addresses were stored exactly as each peer advertised them, including their
  loopback and private network addresses. A node on the public internet then
  retried those every minute forever — and a relay node could end up trying to
  reach a peer by relaying through itself. Remembered addresses are now checked
  before being stored *and* before being dialled, so an existing bad list is
  cleaned up on the next start rather than persisting. Local network addresses
  are still kept, since that is how two machines in one home find each other
  again after a reboot. The per-address log line also moved to debug level; an
  idle node was writing one line per remembered address every minute.
- **A model whose chat template failed to load could be sent an empty prompt.**
  Several kinds of broken template produced no text at all instead of
  reporting an error, and the empty result was used as the prompt — so the
  model received none of the conversation. For image models this also dropped
  the marker saying where the picture belonged. Broken templates are now
  detected and fall back to a sensible format for the model.
- **Vision models with newer image-encoder files failed to load.** Support for
  the two possible layouts is now detected per file instead of assumed.
- **A machine that corrupted shared calculation data was never marked down for
  it.** When several machines split one model layer between them, they combine
  partial results; a machine sending corrupted numbers would spoil the answer
  for everyone in the group, but the fault was recorded as a local problem, so
  its reputation was untouched. It now counts against the machine responsible.
  Cases where the culprit genuinely cannot be identified — a group member
  simply being slow, which might be your own machine — still count against
  nobody, deliberately. Only affects setups that split a single layer across
  machines, which is off by default.

## [0.3.5-alpha] — 2026-07-22

**Five externally-reported bugs fixed (R146) + request cancellation and
deferral cleanup (R147).** All five bugs came from a user running 0.3.4-alpha
on two home machines with an 8B model deliberately split across them. Every fix
below was verified on real GPU hardware, not just unit-tested.

> ### ⚠️ Breaking change — device pools
>
> The pool acceptance signature now covers the invitation's expiry, which is a
> wire-format change. **A 0.3.5 node and a 0.3.4 node will reject each other's
> pool member lists.** If you run a pool, upgrade every device in it together.
> Nodes not using pools are unaffected, and normal inference, discovery and
> credits interoperate across versions as before.

### Fixed

- **A tiny model replicated on two LAN machines could fail outright.** When the
  local node already held every layer it still pulled a LAN peer into a
  tensor-parallel group; if that peer went quiet the whole request died with
  `AllReduce timeout after 10s for layer 0`, on a node that could have answered
  alone. Full local coverage now never forms a group, tensor parallelism is
  opt-in (`inference.tensor_parallel`, default off — over Ethernet the two
  round trips per layer cost more than the compute they split), and a failed TP
  segment now falls back to local compute instead of killing the request.
- **A worker that hit a GPU out-of-memory error kept its VRAM forever.** Only an
  explicit unload ever killed a worker, so a failed request left the process
  resident holding its whole allocation — 4.4 GB in the reported case, still
  there minutes later. Each retry then had less memory than the last, so one
  OOM reliably became permanent failure for that model. Fatal device errors now
  recycle the worker. Measured on an RTX 3070: **VRAM 7951 MB → 120 MB in six
  seconds**, and the retry then succeeds.
- **`inference.gpu_layers` did nothing for sharded models.** It was read by the
  legacy llama.cpp path only; the engine that actually serves inference chose
  its device unconditionally. Setting it to 0, 8 or 20 produced an identical
  allocation. Worse, the shipped default was `0` documented as "CPU only" while
  every CUDA build used the GPU regardless. It is now honoured end to end.
  **The default changes from `0` to `-1` (auto)** so existing GPU nodes keep
  using their GPU; `0` now genuinely means CPU-only, and a node configured that
  way with a GPU present says so loudly at startup. Partial offload is not
  supported by this engine and now warns instead of silently ignoring the value.
- **A GPU OOM now falls back to CPU** for that model for the rest of the run,
  rather than repeatedly retrying the allocation that just failed.
- **Failed requests no longer charge credits for our bugs.** Every failure
  applied a flat −50 penalty, including failures the local node caused —
  debugging the three bugs above drove the reporter's own balance from 0 to
  −470 with no peer ever misbehaving. A penalty now requires that a remote peer
  was actually involved and that the error is one they could have caused.
- **Peer credit balances are no longer invented.** With no gossiped balance the
  leaderboard displayed `trust_score × 5000` — at the default trust of 0.5,
  exactly "+2500 credits" for every unknown peer. This was reported as a ledger
  inconsistency; it was never a ledger figure. Unknown balances now render as
  "—".
- **The dashboard VRAM gauge showed an estimate as if it were live usage.** It
  summed each loaded model's estimated footprint and displayed that instead of
  the real figure, with a clarifying tooltip only when real usage *exceeded* the
  estimate. An idle machine read "5.3 GB / 5.7 GB — 93%" against a real ~1 GB.
  Live usage is now always the headline number; the estimate moved to the
  tooltip.
- **A client that hangs up mid-response now stops the work.** Closing the tab
  left the worker generating its full token budget into a channel nobody was
  reading — measured at 754% CPU thirty seconds after the client died. Now falls
  to 0%.

### Added

- Request cancellation between daemon and worker, covering client disconnects,
  timeouts, and hedge losers. Remote peers are told to stop too, so an enabled
  hedge no longer leaves the losing node computing a result that will be thrown
  away.
- `inference.tensor_parallel` (default `false`).

### Security

- Pool acceptance signatures now bind the invitation's expiry, so a captured
  acceptance is implicitly time-bounded and cannot be transplanted onto a
  different invitation. Expired invitations are also now rejected outright. See
  the breaking-change note above.

### Documentation

- The config reference was missing **45** `[inference]` options; all are now
  documented and grouped (batching, prefix cache, speculative decoding,
  hedging/prefetch, activation transfer). 91 → 133 documented options.
- `swarmpool://` invite codes documented, including why a NAT'd node needs its
  external addresses in the payload.
- Python SDK gained `pool.generate_code()` and `pool.join()`.

1099 → 1124 lib tests, 75 integration tests.

## [0.3.4-alpha] — 2026-07-21

**Cloud model & provider currency refresh + audit sweep (R145).** The cloud
surface hadn't been refreshed in ~2 months. No routing-architecture changes —
providers that fetch `/models` dynamically and route by prefix pick up new
models automatically; only hardcoded lists, aliases, and one base URL were
stale.

- **Claude lineup** — Opus 4.7 → **4.8**, Sonnet 4.6 → **5**, added **Fable 5**
  (Haiku 4.5 unchanged) across the model picker, subscription list, and the
  bare-alias resolver (`opus`/`sonnet`/`haiku`/`fable`). The `claude_subscription`
  default now matches Claude Code 2.1's new default (Sonnet 5). `anthropic-version`
  header confirmed current.
- **Claude Code 2.1.215** — verified every subprocess CLI flag is still valid
  (no breaking removals); added the `manual` permission-mode alias.
- **Moonshot/Kimi** — base URL switched from the China-only `api.moonshot.cn`
  to the international `api.moonshot.ai`; refreshed the model list to Kimi K3 /
  K2.7 Code / K2.6 / K2.5 (the K2-0527 and Moonshot-v1 models were discontinued).
- **DeepSeek** — probe/examples updated to `deepseek-v4-flash` ahead of the
  `deepseek-chat`/`deepseek-reasoner` legacy-name retirement (2026-07-24).
- **Mistral** — added `magistral`/`ministral` prefix routing. **OpenAI** needed
  no change (GPT-5.x and o-series already route correctly).
- **Security** — bumped `anyhow` 1.0.102 → 1.0.104 and `memmap2` 0.9.10 → 0.9.11,
  clearing two RustSec "unsound" advisories surfaced by `cargo audit`.
- Docs refreshed (README, book, config examples). 1099 lib tests, no regressions.

## [0.3.3-alpha] — 2026-07-21

**Dashboard peer-clarity + reachability docs (R144).** Follow-on to 0.3.1/0.3.2,
driven by the first external user testing live — their dashboard called the
remote bootstrap anchor a "LAN" peer.

- **Fixed LAN misclassification** — a peer was tagged LAN if any of its
  *advertised* addresses was private/loopback, but a public `0.0.0.0`-bound node
  also advertises `127.0.0.1`, so every remote peer (including the anchor) was
  mislabeled. Now classified only on the actual connection address + the peer's
  observed-us address.
- **Clear peer typing everywhere** — every peer is tagged **Pool / LAN /
  Internet** (green / purple / blue). The header reads "N internet peers / N on
  your network / N pool devices" instead of an ambiguous "peers / lan". The
  backend exposes a mutually-exclusive taxonomy (pool + lan + remote == connected
  peers).
- **Version in the header** — next to the SwarmLLM logo.
- **Honest empty state** — no longer says "Connecting to the network…" when you
  are already connected with no shared models yet.
- **Swarm-resources strip** — computers online (incl. yours), GPU machines,
  combined VRAM, shared storage, regions — "how big is the swarm actually?".
- **Docs** — README promotes out-of-the-box auto-join (default anchor + UPnP +
  AutoNAT v2 + relay) + Discord; `docs/NETWORKING.md` gained an explicit
  AutoNAT-v2 note.
- i18n: 14 new keys × 21 locales. 1097 → 1099 lib tests.

## [0.3.2-alpha] — 2026-07-21

**Fix: `/dns4` bootstrap peers were undialable.** The swarm transport was
missing the DNS-resolution wrapper (`.with_dns()`), so any DNS-named multiaddr —
including the default `swarmllm.duckdns.org` bootstrap anchor added in 0.3.1 —
failed with "Multiaddr is not supported". Result: fresh installs couldn't
auto-join. Added `.with_dns()`; the default bootstrap also gained an `/ip4`
fallback for hosts where DNS resolution is unavailable. (Caught by live
multi-node validation — the unit test only checked the address string was
present, not that it was dialable.)

## [0.3.1-alpha] — 2026-07-21

**Internet reachability & NAT traversal (R143).** Makes SwarmLLM reachable
across the internet, not just the LAN — the biggest gap for real-world use.

- **Default bootstrap anchor** — fresh installs now dial a publicly-reachable
  seed node (`swarmllm.duckdns.org`) on startup and **auto-join the network
  with zero config**, then decentralized discovery (DHT/PEX) takes over. Set
  an explicit `bootstrap_peers` to override (empty list opts out).
- **UPnP** automatic gateway port-mapping (default on) — zero-config internet
  reachability on cooperative home routers.
- **Invite codes now carry a public address** — `refresh_listen_multiaddrs`
  unions confirmed external addresses (UPnP / AutoNAT / relay / manual) with the
  bound listeners, closing the gap where a NAT'd node minted LAN-only codes.
- **AutoNAT v1 → v2** — v1 falsely reported NAT'd nodes as "Public" over QUIC
  (so they never reserved a relay and stayed unreachable); v2 tests each address
  for real reachability. Plus a belt-and-suspenders relay fallback.
- **`network.external_addresses`** — declare a reachable address (or a list, to
  cover TCP + QUIC) for a port-forwarded box / VPS / dyndns anchor.
- **`--anchor` mode** + a hardened `deploy/anchor/` kit (sandboxed systemd unit,
  non-root user, SHA256-verified binary, firewall, DuckDNS updater) for running
  a public bootstrap/relay node. See `docs/NETWORKING.md`.
- **Security**: quinn-proto → 0.11.15 (RUSTSEC-2026-0185, remote QUIC memory
  exhaustion, HIGH), crossbeam-epoch → 0.9.20 (RUSTSEC-2026-0204); installer
  input validation; relay abuse limits reviewed.
- **Auto-updater tracks pre-releases** — the updater listed `/releases/latest`,
  which skips pre-releases, so alpha nodes never auto-updated. It now lists
  `/releases` and selects by mode: `all` tracks alpha/beta (stays patched during
  the alpha phase), `stable`/`disabled` track stable only. Anchors default to
  `all`.

1075 → 1097 lib tests. Known gap: the relay/DCUtR CGNAT path is wired but awaits
live multi-NAT validation.

## [Unreleased] — post-v0.1.0

Working changelog for commits after the v0.1.0 tag. Will roll into the
next tagged release.

### R142 — Autonomous 8-hour sweep (2026-05-22 → 2026-05-23)

14 sweep rounds dispatched as a self-paced `/loop /sweep`, 15 commits
to `main`, 60+ findings closed. Standout: **3 silent production bugs**
from frontend↔backend JSON wire-format drift — bugs no test caught
because the broken path emits no error:

1. R141 chat empty-state catalog never rendered (WS `stats_update`
   never merged into `App.data.cache.stats` → `cache.stats.wishlist`
   always undefined).
2. R140 maturity-fade button stuck prominent forever (`pool.js` read
   `peer_count` but stats field is `peers`).
3. Auto-manage activity orb permanently zero (`auto-manage-status.js`
   matched PascalCase `'Downloading'`; backend serializes
   `"downloading"`).

Real concurrency bugs: 3 TOCTOU fixed with atomic `remove_if`
(`try_assemble_chunked_forward` double-dispatch, `remove_shard_holder`
holder-drop race, `evict_split_models_lru` cache drift), 2 clock-
dependence bugs (`maybe_reset_window` froze on backward NTP
correction in both hedging + prefetch), atomic ordering on Release-
paired counters, batch `active_count.fetch_add` outside spawn
closure (R103-class leak), `register_multi_turn` orphan
`KvCacheSession`, pipeline-stream `send_forward` orphan on caller-
future cancel, batch panic arm missed `active_pipelines` cleanup.

Security / correctness: HF download size cap bypassed when
`Content-Length` absent (DoS), `auto_switch_quants` Default vs
serde-default mismatch, 4× wrong `SwarmError::Internal` for worker-
died (should be `ServiceUnavailable`), 2 discarded errors that
silently closed SSE streams, `anthropic_split_stream` dropped
`matched_stop_sequence`. Scheduler oracle violation in
`allocate_offline` + mmproj prune missing the same liveness filter.

Hot-path perf: per-token `format!()` in info-level tracing,
`vec![forward.clone()]` deep-copying activation buffer on persistent-
stream non-chunked path.

11 helper extractions, 19 new tests pinning invariants, 4
`.claude/rules/architecture.md` doc drifts fixed, DIAGNOSTICS.md
DIAG strings synced, book introduction + config reference updated.

Deferred to `docs/FUTURE_WORK.md § R142 deferred items`: VLM
`ffn_up/down` weight inversion (needs LLaVA integration test),
LLaVA chat-template fallback edge case, Python SDK R140 endpoints,
test-binary `spawn_test_server` extraction, streaming + invite v2
config-reference doc rows, `apply_update_with_version` Option
cleanup, worker compute waste on cancel.

1056 → 1075 lib tests. Clippy clean default + features
dev,claude-subscription + features llama. Commits:
`05233184..dfcfaa8d`. Detail: `memory/round_log_R142.md`.

### R141 — Auto-manage cold-start UX (2026-05-21)

Closed the "fresh node has nothing to chat with" gap by removing every
silent gate that blocked auto-manage from acting and surfacing what
the swarm already runs directly in the chat empty state. Coordinated
six changes:

- **Trusted-publisher allowlist** in `src/model/huggingface/watcher.rs`.
  `TRUSTED_HF_PUBLISHERS` covers official model authors (meta-llama,
  mistralai, Qwen, google, microsoft, deepseek-ai, HuggingFaceH4,
  stabilityai, tiiuae, 01-ai, NousResearch, allenai, ibm-granite,
  CohereForAI) + curator community (bartowski, TheBloke, unsloth,
  lmstudio-community, MaziyarPanahi, QuantFactory, second-state).
  Models from trusted publishers promote to `DemandVerified` at 10k
  HF downloads instead of 100k — fresh releases from known curators
  reach the swarm in hours not weeks. 24h age gate unchanged.
- **Wishlist `Candidate` status** in `src/model/auto_manage/wishlist.rs`.
  `compute_wishlist` now merges HfTrending entries the swarm hasn't
  adopted (cap 24) as `Candidate` rows with new `hf_repo_id` +
  `task_tags` fields. Frontend renders these with a "Set this up"
  CTA that opens the HF browse pre-filtered to the repo — user
  picks the quant variant, no auto-pick. +10 score bonus for trusted
  publishers.
- **`auto_switch_quants` default → `true`** in `src/config/inference.rs`.
  Recommendation surface that required a button-click became automatic.
  Trust + prune cooldown still guard bandwidth cost. Operators on
  metered links can flip back off.
- **`P2P_PERMIT_STALL_SECS` 600 → 180** in `model/auto_manage/manager.rs`.
  Silent libp2p drops fail over to HF fallback in 3 min, not 10.
- **`activity.hf_sources_cap_reached` activity event** in
  `daemon/dispatch/mod.rs`. Throttled 1st + every 50th drop, warning
  toast pointing the user at Settings cleanup. Previously silent
  `tracing::warn!`-only.
- **Chat empty state swarm catalog** — `createEmptyState` in
  `frontend/js/core/utils.js` builds a `buildSwarmCatalog()` block
  when no model is selected. Three rows: Serveable (one-click select),
  Aspirational (gathering), Candidate (route to HF browse). Chip click
  handlers route through `App.models.selectDropdown` +
  `App.chat.newSession` so the user lands in a fresh chat with the
  model loaded in one click. Re-rendered on every `stats_update` so
  the catalog comes alive within ~2s of daemon start.

i18n: 15 new keys × 21 locales (1156 → 1172 entries per locale) —
idiomatic translations. Tests: +5 watcher + +3 wishlist. 1048 → 1053
lib tests (R140 had brought it to 1048). Clippy clean default + features
dev,claude-subscription + features llama. Commit: `50225f7c`.

### R140 — Pool invite codes v2 (2026-05-19)

The 8-character pool invite code (`A3F7K2M9`) worked only when both
nodes were already on the same libp2p swarm — useful in a mature
decentralized network, useless for helping two fresh nodes find each
other before decentralization is achieved. R140 closed that gap.

New `swarmpool://...` blob (`src/pool/invite.rs`) wraps the existing
8-char code with the inviter's reachable listen multiaddrs. Encoded as
JSON → ChaCha20-Poly1305 (random embedded key, anti-IP-harvesting
only) → base64url. ~300-500 chars, fits in a copy-paste. Inner
payload: `{ version, pool_id, pool_name, multiaddrs[], code (8-char),
expires_at_unix }`.

- New `SharedState.listen_multiaddrs: ArcSwap<Vec<String>>` — live
  snapshot rebuilt by NetworkManager on `NewListenAddr` /
  `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed`.
  Filtered via `addr_is_remotely_reachable` that drops loopback,
  unspecified, link-local, and AWS IMDS but keeps Tailscale CGN
  (100.64.0.0/10) — the WAN-bootstrap use case needs that range.
- `handle_join_with_code` dual-mode: v2 blob → dial each multiaddr
  via `NetworkCommand::DialAddress` then broadcast existing
  `PoolMessage::JoinRequest`. Legacy 8-char → direct broadcast
  preserves on-swarm flow. Wire protocol unchanged.
- Generation rejects empty addresses: if `listen_multiaddrs` is empty
  (daemon hasn't bound yet), `handle_generate_invite_code` returns
  `ServiceUnavailable` instead of handing out a useless code.
- Frontend: dropped the fake-QR pattern (only hashed 8 chars, was
  never scannable — misleading). Monospace code box + Copy button
  sized for ~500-char v2 blob. Paste field upgraded from `<input
  maxlength=8>` to `<textarea>`. Join handler sniffs prefix to route.
- Maturity-fade UI: while local node sees <50 swarm peers, "Add
  Another Device" sits in the dashboard header. ≥50 peers demotes it
  to Settings — swarm is mature enough that DHT discovery is reliable.

5 i18n strings refreshed × 21 locales + 1 dead key removed (`pool.
scan_or_type` — for the fake QR). 18 new tests (codec roundtrip,
tamper, expiry, version, truncated/oversized, prefix sniff; 3 listen-
addr filter; 5 PoolManager paths). 1030 → 1048 lib tests. Commits:
`c49632af..d7d77e6e`.

### R139 — Tier 4K communication-computation overlap (2026-05-19)

Five commits closing FUTURE_WORK Tier 4K with a research-driven scope
pivot. Phase B turned out to already be shipped via existing async
architecture; documented and skipped. Original "worker streams row-
tiled output during matmul" pivoted to **daemon-side STREAM-chunked
encrypt+send** on a single libp2p stream (age STREAM construction +
TokenWeave K=2-4 sweet spot + Tink Streaming AEAD precedent) after
research found no production inference system streams forward-output
tensors (Triton decoupled, vLLM v1, NVIDIA Dynamo/NIXL — all single-
tensor responses).

- **Phase C** (`11333f67`) — ChaCha20-Poly1305 encrypt/decrypt
  offloaded from NetworkManager event loop. CPU-bound sealing in
  `handle_send_tensor` and the open in `handle_tensor_payload`
  (TENSOR_TAG_ENCRYPTED arm) now `tokio::spawn` tasks. New
  `NetworkCommand::SendEncodedTensor` carries the encrypt-result
  back. ~50-200µs/forward event-loop savings; under concurrent
  decode this is the difference between smooth event-loop
  responsiveness and observable jitter on libp2p ping / gossip /
  connection events.
- **A-rev.1** (`4b5fc10c`) — Wire-format `ChunkMeta { chunk_idx,
  total_chunks }` trailer 0x05 + AAD binding via
  `build_layer_forward_aad`. Reorder, wrong-total, and cross-transfer
  substitution all fail Poly1305 before reaching dispatch. 11 new
  tests.
- **A-rev.2/3** (`1d0a5d55`) — Receiver assembly on
  `SharedState.pending_activation_chunks: DashMap<Uuid,
  ChunkAssemblyState>` (root-level). `chunk_layer_forward` splits at
  byte offsets; passthrough when activation ≤ chunk_size. 4 config
  knobs (`streaming_chunked_send`, `streaming_chunk_size_bytes`,
  `streaming_min_activation_bytes`, `streaming_chunk_assembly_ttl_secs`),
  default off.
- **A-rev.4** (`e32c0a5d`) — Sender wired in
  `pipeline/distributed.rs::forward_through_segments` persistent-
  stream path. RR fallback intentionally NOT wired (needs per-chunk
  Acks — deferred).

1015 → 1030 lib tests (+15) + 4 new swarmllm-types tests. Clippy
clean default + features dev,claude-subscription. Commits:
`11333f67..1108c817`.

### R138 — Autonomous defer-batch sweep-log triage (2026-05-18)

Eight commits closing ~20 deferred sweep-log items via 8 real fixes +
~15 verification-only entries:

1. **Auto-manage rescan respects `auto_manage_paused`** (R104 closure).
   Rescan still runs locally (correctness — picking up manually-
   placed shards) but the network re-announce gates on
   `auto_manage_enabled`.
2. **`active_count.fetch_add` inside spawn closure** (R103 closure).
   No leak on `tokio::spawn` OOM panic.
3. **`CreditBalance` `#[serde(default)]` + doc convention** (R105
   closure). Forward-compat for schema upgrade; node never silently
   restarts at zero balance on field addition. `swarmllm-types` got
   a `[dev-dependencies] serde_json` so 15 previously-dead lib tests
   now run.
4. **`private_mode`/`offline_mode` moved out of `pool_state` tree**
   (R105 closure). New `TREE_NODE_MODES` + `restore_node_mode()`
   migration helper. Each tree single-typed.
5. **`check_integrity` strict per-tree type validation** (R105
   closure). `validate_strict` routes each `CRITICAL_TREES` entry
   through the actual `swarmllm_types` type. Type mismatches that
   passed JSON-Value validation are now flagged corrupt.
6. **`credit_percentile_cache` no longer held across DashMap iter**
   (R97 closure). Three-phase pattern; router task no longer blocks
   on long iters.
7. **`api.metrics_auth_required` config flag** (R101/R102 closure).
   Tightens `/metrics` for public-internet nodes by removing the
   loopback exemption.
8. **Credit forward per-window value cap** (R102 closure).
   `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW = 200k` credits/min/member
   on top of the existing count cap.

Plus ~15 verification-only sweep-log closures for items intervening
rounds had already addressed (R66/R67/R68/R89/R97/R102/R103/R104/R105/
R123). 1005 → 1015 lib tests (+10) plus 15 newly-runnable
`swarmllm-types` tests. Clippy clean default + features dev,claude-
subscription + features llama. Commits: `d122e9e8..cb60b2ed`.

### Sweep arc R122 → R124 (2026-05-12 → 2026-05-13)

Three rounds of standard-rotation sweeps after R121 closed. R122 and
R123 found 28 fixable findings; R124 stalled (worktree agents hit the
10-min watchdog) and was closed without findings — diminishing returns
on top of the R101-R109 security arc and the R110+ structural work.

**R122 (commit b530c42b) — 22 findings auto-fixed**

- 20 SwarmError variant fixes across `src/inference/process_pool.rs`
  (14 sites — model-worker subprocess lifecycle: spawn, IPC connect,
  socket bind, send Forward/BatchForward/Generate, worker-dead),
  `src/update.rs` (4 sites — apply-update file operations),
  `src/api/admin_providers.rs:1029` (current_exe), and
  `src/api/claude_session/handlers.rs:166` (create_dir_all temp).
  All previously surfaced HTTP 500 ("internal error") for subprocess /
  OS-level failures that should be 503 ("service unavailable"). Same
  pattern R118-R120 fixed for `claude_sub.rs` and
  `claude_session/manager.rs`. The `get_or_spawn` slow path already
  used `ServiceUnavailable`, making the inconsistency visible.
- `src/inference/tensor_util.rs:11-12` — `DTYPE_TAG_F32` /
  `DTYPE_TAG_Q8_0` had `pub` visibility but only in-file callers.
  Tightened to module-private `const`.
- `frontend/js/components/notifications.js` — three call sites
  inlined `500ms / 2000ms` latency-tier thresholds with two distinct
  class systems (`dot-*` vs `health-fast/ok/slow`). Extracted to
  `_dotLatencyTier()` and `_healthLatencyTier()` helpers.
- Doc drift: test count 909→913, i18n key count 1122→1130 (post-R121).

**R123 (commit d2a876bf) — 6 findings auto-fixed; 3 false-positive
i18n orphans caught**

- `src/api/admin.rs:362` — `update_config` silently mapped any
  unknown contribution-mode string to `Moderate` via `_ => `. Now
  returns 400 Validation per the contract.
- `src/api/admin.rs:471` — config-save `std::fs::write` OS failure
  promoted from `Internal` to `ServiceUnavailable`.
- `src/api/anthropic/handlers.rs` — two near-identical
  `MessagesResponse` builders extracted to
  `fn build_messages_response(request_id, model, output) ->
  MessagesResponse`. Third call site (legacy executor in
  `mod.rs:245`) uses a different result type and stays inline.
- `frontend/js/core/utils.js:411` — duplicate active-state predicate
  now defers to `App.downloads.isActiveDlState`.
- `docs/ARCHITECTURE.md` — `ModelMgmt` sub-struct diagram missing
  `contribution_auto` (R121); `dashboard-shards.js` listed twice in
  JS component list. Both fixed.
- **Verification gating caught 3 false positives**: Agent 4 flagged
  `settings.section_contribution`, `section_identity`, and
  `section_preferences` as orphaned i18n keys; the required cross-grep
  (`frontend/js/`, `index.html`, `css/`) found `data-i18n` callers
  in `index.html:302/245/270` for all three. Logged as wontfix so
  next sweep doesn't re-claim them.

**R124 (commit 66426fb4) — inconclusive**

Three specialised agents (concurrency + hot-path, security + authz,
Rust idioms + perf) all stalled at the 600s worktree watchdog. Closed
without findings; the deeper categories the round targeted (security,
concurrency) were already drilled hard in R101-R109 (44 fixes), so the
miss is not load-bearing. Re-run if a specific concern surfaces.

**Stats**

- 28 findings auto-fixed across R122-R123 (22 + 6)
- 3 false-positive i18n orphans caught by R120 verification rule
- 5 deferred (separator divergence, country-names i18n, etc.) —
  logged in `.claude/sweep-log.jsonl`
- sweep-log.jsonl grew 1131 → 1173 (+42 entries)
- 913 lib tests pass; clippy clean default + dev features

### R121 — Auto-manage scale-back at swarm saturation (2026-05-12)

Auto-manage learned to scale a node's contribution DOWN, not just up.
At swarm scale (1000s of nodes), a popular model is held by far more
peers than the geo-aware target needs — an idle node's shards become
redundant and just waste VRAM. R121 lets auto-manage shed those shards
voluntarily, without waiting for VRAM/disk pressure to build.

**New config field** — `[node] contribution_auto: bool` (default `true`).
Auto mode lets auto-manage scale contribution up AND down within the
user-set `[node] contribution` cap and `[resources]` caps. Manual mode
(`false`) pins contribution at the user-set level — pre-R121 behaviour.

**Saturation-aware prune.** `model/auto_manage/prune.rs` gains
`effective_prune_target(target, pressure, holder_count, contribution_auto,
min_replicas)`. When `contribution_auto` is true AND
`holder_count >= 1.5 × target`, the function bypasses the RELAXED-state
+1 nudge from `pressure_adjusted_target` and uses the raw target — so
the shard is eligible to prune even at zero local pressure. Severe
saturation (`holder_count >= 2 × target`) gets a flat +1.0 prune-score
bonus to break ties. All existing prune guards still apply
(active-pipeline, pinned/locked shards, configured-range, would-eliminate-
region, can-reacquire, recently-acquired, encrypted-pipeline models).

**Hot-reload.** `state.models.contribution_auto: AtomicBool` mirrors the
config field. `PUT /api/admin/config` updates the atomic so the toggle
takes effect on the next prune tick without a daemon restart. `state.config`
remains startup-frozen — the atomic is the only runtime source of truth
for the toggle, and the field is documented in `.claude/rules/architecture.md`
as a SharedState invariant.

**API.** `ConfigUpdate` and `GET /api/admin/config` gain
`contribution_auto: bool` and `max_gpu_vram_mb: u64`. The latter was
previously in `[resources]` config but not in the API — users had to
edit TOML and restart to set a VRAM cap. The VRAM cap is hard-capped
at 1 TiB on PUT to prevent UI typos from disabling VRAM accounting.
`OperationalParams` gains `contribution`, `contribution_auto`,
`max_gpu_vram_mb` for completeness (broadcast on hot-reload).

**Frontend.** Auto/Manual toggle in Settings panel above the existing
contribution segmented control. In Auto mode, "Contribution Level"
relabels to "Upper Cap" and the hint explains the scale-back semantics.
Wired via `App.settings._applyContributionMode(modeAuto)`. New i18n
keys: `settings.contribution_label_cap`, `settings.contribution_mode_*`
(8 keys total) translated across all 21 locales by translator-agent.

**Tests.** 4 new unit tests in `prune::tests` cover the saturation
override at boundaries (not-saturated/fall-through, saturated/no-pressure,
severe-saturation, min-replicas floor). All 913 lib tests pass; clippy
clean default + `--features dev,claude-subscription`.

**Deferred to a follow-up.** Setup wizard exposes the toggle (the wizard
already has the segmented contribution control + auto-manage checkbox;
the toggle slots in but is additive). Holder counts at >50 use the
gossip-cached value rather than a separate uncapped DHT-provider count
— at realistic targets the 50-cap is well above SATURATION_FACTOR×target,
so prune still fires correctly, only the displayed redundancy_ratio
underestimates how aggressively to prune. Both items captured in
`docs/FUTURE_WORK.md`.

### Security & stability sweep arc R92 → R109 (2026-05-01 → 2026-05-08)

Eighteen rounds of security and stability sweeps, ~150 fixes total.
Highlights: 14 HTTP/P2P/crypto vulnerabilities (R101), 17 concurrency/
economic/DoS findings including a CVE-adjacent issue (R102), 13 authz/
numeric/cancel/CI fixes (R103), 10 auto-manage audit fixes + shard-
reality messaging (R104), 11 inference math + distributed pipeline +
DB + HF + metrics fixes (R105), 17 logprobs/oracle/tier-bypass/spec-
DSD bookkeeping fixes incl. dedup helpers (R106), 9 tier-bypass/weak-
ordering/DoS-cap/crash-ordering fixes (R107), 9 Qwen3.5 KV mask + API
parity + hot-path perf fixes (R108), 3 R108-deferred completions (R109).
AAD now covers spec/kv-truncate trailer fields. Per-page bootstrap
nonce on `/api/admin/api-key`. MCP method-string reflection cap.

### Model management redesign — R110 → R116 (2026-05-08 → 2026-05-10)

User-visible swarm and model UX rebuild for non-technical audiences:

- **R110 — swarm-capacity foundation.** New `state.metrics.swarm_capacity`
  (ArcSwap<SwarmCapacity>) snapshot eagerly refreshed on peer connect/
  disconnect so the dashboard banner stays consistent with the peer
  panel under churn (the 1.5s stats-cache coalesce alone was too lazy).
  Makes collective swarm power visible.
- **R111 — wishlist subsystem.** `state.models.wishlist` shows users
  what auto-manage is actually planning. Surfaced through a new Swarm
  tab (`frontend/js/components/swarm-tab.js`) with wishlist + Capacity
  Plan views.
- **R112 — HfWatcher.** New subsystem (now 12 Tokio tasks; was 11). Hourly
  HuggingFace trending-GGUF poll seeds the wishlist and auto-promotes
  models above download/age thresholds to `DemandVerified`.
- **R113 — Capacity Plan / What-If.** Turns "contribute" from abstract
  to concrete: shows what specific shards your VRAM could host.
- **R114 — HF browse polish.** Task filter chips + status-driven CTAs.
- **R115 — onboarding storage preview** and i18n parity audit.
- **R116 — audit closure.** R110-R115 i18n translations finalized
  across all 21 locales (1122 keys / 1124 entries per locale, native
  language strings rather than English fallback). Active-pipeline
  guard added to `delete_model`/`delete_shard` admin handlers — yanking
  a shard mid-token-loop now returns 503 instead of corrupting state.
- **Dashboard rework.** Network Status panel + Models tab + inline HF
  browser. Removed the legacy HF download modal.

### Sweep arc R117 → R120 (2026-05-10 → 2026-05-12)

Four sweep rounds, 22 findings auto-fixed:

- **R117** — `spawn_check_and_load` helper consolidates the "shard
  landed → reload model → refresh dashboard" pattern across three call
  sites. `SystemContent::to_plain_text` dedup. `browse.*` translations
  added to all 20 non-English locales. Modal-only dead code dropped
  from HTML/CSS/JS/i18n.
- **R118** — 7 findings: error variant misuse cleanup (subprocess paths
  Internal → ServiceUnavailable), input validation cap on `tasks`
  query param, dead `#[allow(dead_code)]` attribute removed, doc drift
  fixes (HTML template count 13→11, dead "provider badges" prose
  removed).
- **R119** — 5 findings: dead pub visibility tightened (`MAX_TRENDING_
  ENTRIES`), 7 subprocess sites Internal → ServiceUnavailable in
  `claude_sub.rs`, 3 `write_to_stdin` sites in `claude_session.rs`,
  `openai/responses/translate.rs` upstream parse failures
  Internal → ProviderError {502}. Dead `App.hf.download` removed with
  4 orphaned i18n keys cleared across all 21 locales.
- **R120** — 5 findings, 1 wontfix: more dead pub visibility tightening
  (`coalesce_byte_ranges`, `cross_node_prefix_holders`, `EVENT_BUFFER_
  CAP`). MCP `tool_delegate` replaced 70-line inline /v1/messages
  dispatch with `dispatch_model_call` helper. Verification gating
  caught 6 false-positive i18n orphan claims (callers existed in
  init.js/utils.js/chat.js) — logged as wontfix instead of deleted.
- **completeness.md** codified the new sweep rules: "Verify before
  deleting sweep findings" and "Re-exports and visibility downgrades".

909 lib tests + 75 integration tests passing at the end of R120; clippy
clean on default + `--features llama`. CI now ignores 5 known accepted
RUSTSEC advisories (added RUSTSEC-2026-0097 ignore to match
`SECURITY.md` table; rand custom-logger advisory is warning-level
without a custom logger but is now explicitly ignored for parity).

### Sweep arc R76 → R81 (2026-04-29 / 2026-04-30, autonomous)

Self-managed overnight sweep covering: doc drift + dead code (R76),
concurrency + lifecycle (R77), error paths + resilience (R78), security
+ wire format (R79), frontend (R80), hot-path performance (R81). Each
round spawned 3–4 parallel review agents, applied auto-fixable
findings in batched commits. ~7 commits.

- **R76 (doc drift + dead code)** — Removed unused `SplitModel::n_kv_head()`
  accessor (no call sites). Fixed `request_response` timeout doc claim
  300s→600s. Clarified `GossipSub mesh_outbound_min` scales with peer
  count (1→4 across 6 buckets) instead of fixed=1. Corrected several
  ARCHITECTURE.md path references (`pipeline/distributed.rs`, not
  `pipeline/mod.rs`; `state/activity.rs` for ActivityEvent struct).
  i18n key count 1014→1015 (verified). CLAUDE.md Bronze tier:
  "negative balance" → "zero/negative" to match `priority.rs`.
  Replaced bare `#[allow(clippy::excessive_precision)]` on CLIP
  constants with rationale comment.
- **R77 (concurrency + lifecycle)** — Five concurrency bugs:
  - `dispatch/mod.rs:880` pool_tx RwLock guard held across send().await
    (could deadlock dispatcher AND any installer of pool_tx). Clone
    Sender out of guard before send.
  - `dispatch/mod.rs:261` per-peer LayerForward count load-then-add
    race (admits MAX+1 forwards from one peer). Switch to optimistic
    fetch_add → check prev → fetch_sub on overshoot.
  - `dispatch/mod.rs:456` router_tx send().await blocking the dispatch
    loop on a backlogged InferenceRouter. Switch to try_send.
  - `dispatch/mod.rs:402` pending_vision_results get-then-remove
    TOCTOU vs health-monitor's stale-entry sweep. Atomic remove +
    re-insert on sender mismatch.
  - `process_pool.rs:905` spawn_failures cooldown bypass when fresh
    failure recorded between (clear) check and spawn_lock acquisition.
    Move cooldown check inside the lock.
  - `network/manager/requests.rs:398` peer_to_node Ref held across
    peer_registry.get_mut (gotcha #10 pattern). Clone NodeId out of
    Ref before second DashMap access.
- **R78 (error paths + resilience)** — Wrong error variants and
  silent failures:
  - `inference/router/mod.rs:285` emit `InsufficientCredits {balance,
    required}` (→402) instead of `CreditError(string)` (catch-all
    →500). Same fix on queue-full ServiceUnavailable.
  - `error.rs:203` `SwarmError::Config` maps to 500/server_error,
    not 400/invalid_request_error (rule: Config is startup-only).
  - `chat_template/eval.rs:284` guard `% 0` modulo. Peer-supplied
    GGUF chat templates can crash the worker subprocess every request.
  - `chat_template/eval.rs` depth-limit recursion (Cell<u32> + RAII
    DepthGuard, MAX_TEMPLATE_DEPTH=256). Prevents stack overflow
    from `((((...))))` or nested for-in-for templates.
  - `network/manager/tensors.rs:212` gate per-tensor-forward
    connection_addrs Vec<String> dump on tracing::enabled!(DEBUG).
    At default info level the eager allocation burned ~50–100 KB
    throwaway heap per LayerForward.
  - `model/acquisition.rs:344` set `AcquisitionState::Failed` when
    manifest save fails (was leaving the dashboard stuck on
    Downloading forever).
  - `api/admin_models/lifecycle.rs:58` log DB::remove errors instead
    of `_ = ...`. Asymmetry vs file-removal path caused divergence
    after partial deletes.
  - `update.rs:97` log warn when `current_exe()` falls back to
    "swarmllm" relative path (sandboxed envs).
- **R79 (security + wire format)** — Replay protection, input
  validation, IPv6 CORS:
  - `dispatch/mod.rs:808` (NicknameGossip): one-sided staleness per
    gotcha #44. Future-dated records were squatting peer nicknames
    for up to 24 hours.
  - `dispatch/mod.rs:518` (CreditTransaction): freshness window
    (30s skew, 5min max age). Was admissible indefinitely.
  - `inference/split/gguf_meta.rs`: cap block_count, embedding_length,
    head_count, head_count_kv at sane upper bounds (256 / 65 536 /
    256). Crafted GGUF was driving worker into oversized KV/mask
    allocations.
  - `pool/manager/gossip.rs:286`: cap inbound device_name to 64 bytes.
    Local handler caps at 32 chars but inbound gossip path was
    persisting multi-MB strings to all pool members' redb.
  - `api/mod.rs`: add `validate_optional_sampling` for top_p,
    top_logprobs, presence_penalty, frequency_penalty (was silently
    clamped inside build_sampling_params, violating OpenAI-compat
    spec contract).
  - `api/openai/mod.rs`: cap response_format.json_schema.name (256B)
    and .schema (64KB). Bypassed validate_content_size.
  - `middleware.rs` cors_layer + `websocket.rs` origin allowlist:
    add `http://[::1]:{port}` for IPv6-only browsers.
- **R80 (frontend)** — Two real bugs:
  - `dashboard.js`: ResizeObserver/IntersectionObserver attached to
    .shard-matrix per model card stayed live after `list.innerHTML=''`
    wiped the DOM, accumulating across each `models_changed`
    re-render. Add `_disconnectMatrixObservers()` called before each
    of the 3 wipe sites.
  - `chat.js`: `saveSessions()` now surfaces QuotaExceededError as a
    warning toast instead of silently dropping chat history. New
    i18n key `chat.storage_quota_exceeded` across all 21 language
    files (1015→1016 keys, 1017→1018 entries per locale).
- **R81 (hot-path performance)** — Three measurable wins:
  - `pipeline/distributed.rs:787` per-token info!→debug! for the
    `Sending LayerForward to remote segment` log. ~4 String allocs/
    token/segment saved at default level.
  - `split/executor.rs:278` gate the unconditional `Instant::now()`
    on `tracing::enabled!(DEBUG)`. clock_gettime syscall per forward
    eliminated when info-only.
  - `router/mod.rs` hoist `chatml_fallback()` out of the if-let so
    the same prompt String is reused by both
    `check_multi_turn_reuse` and `register_multi_turn` on the
    cache-miss branch.

Out-of-band fix: formalized the `tests/fixtures/tiny_model/` fixture
deferral (Option B) — false aspirational claim removed from CLAUDE.md
and `local_embedder.rs` test gated on `SWARMLLM_TEST_MODEL_DIR` env
var pointing at a real on-disk model dir. New entry under § Deferred
Items in `docs/ARCHITECTURE.md` explains why a synthetic random-weight
GGUF would only catch parser/IPC plumbing bugs already covered by
unit + in-process tests.

### Sweep arc R82 → R89 (2026-04-30, autonomous)

Continuation of the overnight sweep arc, six new rounds plus a follow-up
deferred-item batch. ~17 commits. Each round spawned 3–4 parallel
review agents and applied auto-fix findings as they landed.

- **R82 (RAII guard scope)** — Document why
  `pending_layer_results` RAII guard cannot be applied in
  `forward_through_segments`: borrow conflicts with later `&mut self`
  segment calls would force an unwieldy lifetime gymnastics. Note in
  source comment + sweep log; the sibling `failover_segment` IS
  patched in R85.
- **R83 (correctness)** — Prune race window in `auto_manage/prune.rs`
  (compute redundancy, decide to prune, but didn't re-check between
  the decision and the file rm; R83 atomic-marks as pruning before
  fs::remove). Escrow lock pattern aligned with credit-tx pattern
  (apply balance change first, then audit log). Smaller fixes:
  `unwrap()` paths reachable on bad input, redundant `clone()` in
  hot scoring loop.
- **R84 (pool gossip + error types + i18n)** — Pool gossip caps for
  `DeviceStatsReport.device_name`/`models_hosted`/`model_name_len`
  (multi-MB strings could otherwise be smuggled into every peer's
  pool state via inbound gossip). Two error-type fixes:
  `SwarmError::Internal` → `Validation` for API input rejections,
  `SwarmError::Config` → `Internal` on a runtime-only path. i18n
  count-interpolation pattern (`{count}` placeholder) replacing
  ad-hoc `' ' + n + ' '` string concat.
- **R85 (security hygiene)** — Zeroize on drop for
  `crypto/session.rs::CachedSession` and provider API keys
  (`crypto/provider_keys.rs`) so process memory dumps don't leak
  recently-rotated session keys / tokens. Also: failover path
  `pending_layer_results` leak (gotcha #70). Pattern lifts the RAII
  guard from gotcha #45 — clean up the slot on `wait_for_result`
  Err path so a double-timeout (primary + standby both unreliable)
  doesn't permanently deplete `MAX_PENDING_LAYER_RESULTS=1024`.
- **R86 (dispatch + cleanup)** — Shard continuation orphan
  (`pending_shard_continuations` map grew without cleanup if the
  client disconnected mid-stream; TTL sweep added). Dead `NodeStats`
  fields removed (`bytes_uploaded`/`bytes_downloaded` superseded by
  `shared_state.shard_bytes_served` atomic). Two flaky tests
  stabilised. Python `swarmllm-client` stats endpoint mirrored the
  R67 atomic split. **Follow-up**: shard-serve disk I/O + bandwidth
  throttle moved off the swarm event loop (gotcha #11 / #71). The
  inbound `SwarmRequest::ShardTransfer` handler had been doing
  `read_shard_chunk_async()` + `tokio::time::sleep()` (bandwidth
  cap) inline; at a 1 Mbps cap with 4 MB default chunk, the loop
  froze ~32 s. Now stashes the `ResponseChannel` in
  `pending_shard_responses` keyed by ticket and `tokio::spawn`s the
  I/O+throttle, mirroring the `PrefixKvFetch` pattern.
- **R87 (cancel observation + dispatch hardening)** — Cancel signal
  added to four fast paths that had forked from `execute_distributed`
  but omitted the cancel observation (gotcha #72: `local_exec`,
  `try_speculative_distributed`, `try_dsd_distributed`,
  `try_remote_generate_fastpath`). Dispatch hardening:
  - one-sided u64-millisecond gossip timestamp checks for
    `RegionShardSummary` and `ModelDemandGossip` (gotcha #73,
    extends gotcha #44 — `saturating_sub` returns 0 when `ts > now`,
    so future-dated messages bypassed the staleness gate);
  - dispatch loop sub-system sends switched from `.send().await` to
    `try_send` for `pool_cmd_tx` (gotcha #74 — the dispatch loop is
    the network event loop's only consumer, blocking it on a slow
    PoolManager starves every other inbound message);
  - cap raises for inbound caps that had been at TinyLlama-era
    sizes.
- **R88 (auto-update default + credit invariants)** — `AutoUpdateMode`
  default flipped `Stable` → `Disabled` (gotcha #75). Until binary
  signing (C1) lands, `src/update.rs` only verifies a SHA256 sidecar
  fetched from the same release as the binary; a compromised
  maintainer/CI token can publish a matching pair. Defaulting to
  Stable was silently downloading unsigned binaries on every node's
  startup. `apply_credit_direct` now reverts in-memory mutation on
  DB persist failure (gotcha #76 — the silent double-credit was
  undetectable from logs). Rebalancer cooldown scope tightened
  (was per-process when it should have been per-(model, target-region)).
  HTTP error-mapping fixes for two paths.
- **R89 (deferred-item follow-ups)** — Three deferred items closed:
  - HF download size cap + vision overflow guard + IPC header cursor
    fix (`147f474`).
  - **OpenAI `frequency_penalty`/`presence_penalty`** implemented per
    OpenAI spec (gotcha #77): penalty applied *before* temperature
    scaling, threading `generated_ids` through standard worker decode
    + batched decode + distributed-path `LayerForward.generated_ids`
    / `IpcForward.generated_ids` (`#[serde(default,
    skip_if=is_empty)]` so zero-penalty requests stay on existing
    wire shape). Only the LAST segment of `forward_through_segments`
    samples; intermediate segments don't.
  - **Auto-compute BLAKE3 for zero-hash shards + mmproj**
    (gotcha #78). `auto_manage/scan.rs::rescan_local_shards`, when
    seeing a manifest shard with `hash == [0u8; 32]` whose file
    passes the size check, computes BLAKE3 in `spawn_blocking`,
    persists, THEN registers as holder. Without this, manually-
    placed shards with zero-hash placeholder manifests would be
    announced to the network using only a size check. mmproj path
    in `auto_manage/download.rs::trigger_mmproj_download` does the
    same.

### Sweep arc R90 → R91 (2026-04-30 / 2026-05-01, autonomous)

Two more rounds, 25 fixes pushed. Continued the same pattern of 4
parallel review agents per round with worktree isolation.

- **R90** (`50af1b8`, 12 fixes) — Hot-path safety, dedup helpers,
  doc drift:
  - **Dead `updateShardsLive` summary block** in `dashboard.js`
    referenced 6 undefined counters and a `data-model-summary`
    element that no longer exists (replaced by torrent-style health
    bar in commit `8ac662c`). Removed the block + 6 orphan i18n keys
    across all 21 locales.
  - **`finish_speculative` inline copy** in `dsd.rs` replaced with
    method call (already on `PipelineExecutor`).
  - **`build_layer_forward_aad`** extracted to
    `network/protocol/encrypted.rs` so encrypt-side
    (`network/manager/tensors.rs::handle_send_tensor`) and decrypt-
    side (`decode_layer_forward_encrypted`) compute the AAD bytes
    from a single function. Drift between the two would silently
    break every encrypted forward — gotcha #80 pins the contract.
  - **`gossip_timestamp_fresh`** extracted in
    `daemon/dispatch/mod.rs` — the gotcha #44 one-sided staleness
    check was duplicated for `RegionShardSummary` and
    `ModelDemandGossip`. Now centralised; gotcha #81 records the
    helper as the canonical entry point for new gossip types.
  - **`MAX_RESPONSES_INPUT_ITEMS = 1024`** + per-`InputMessageItem`
    extras enforcement added to `validate_responses_ingress`. Closes
    a DoS surface where a request with thousands of message items
    each carrying 32 × 4 KB extras could bypass the top-level
    `extras` cap.
  - **DIAG demoted** `info!` → `debug!` for `build_prompt from
    header` (was firing on every prefill at default log level).
  - **`REMOTE_GENERATE_TOKEN_CHANNEL_CAP = 256`** named constant
    replacing magic literal.
  - **`/api/admin/hf/download` deprecation note** in
    `docs/ARCHITECTURE.md` — frontend MUST use `/download-shards`
    per CLAUDE.md "no implicit full model downloads" rule.
  - **`update.checking` i18n key** added across all 21 locales (the
    update download button was reusing `settings.detecting`, which
    is semantically hardware detection — confusing in the update
    context for users reading the translation literally).
  - **Doc count corrections** — lib tests 897 → 887 in CLAUDE.md
    Testing + Status sections; i18n keys 1030/1032 → 1025/1027 in
    CLAUDE.md + ARCHITECTURE.md.
- **R91** (`9993960`, 13 fixes) — Visibility, magic-number hoists,
  freshness dedup:
  - **Visibility narrowed** `pub(super)` → `fn` on four
    internal-only helpers: `eligible()` in `speculative.rs`,
    `dsd.rs`, `remote_generate.rs`, and `argmax()` in
    `speculative.rs`. None had cross-file callers.
  - **R90 regression caught** —
    `speculative.rs::finish_speculative` needed `pub(super)` for the
    cross-file call from `dsd.rs` introduced in R90's inline-copy
    removal. Default-feature `cargo check` skipped the llama path
    that triggered the breakage. Gotcha #79 records this footgun:
    visibility-tightening or cross-file refactors that touch the
    spec/dsd modules MUST verify with `cargo check --features
    llama`.
  - **Non-llama `DraftState` stub** had an unused `pos: usize` field
    guarded by `#[allow(dead_code)]`. Per
    `.claude/rules/completeness.md` `#[allow(dead_code)]` is a
    smell, not a fix. Converted to a unit struct (the stub is never
    constructed in non-llama builds).
  - **Hoisted 8 block-scoped consts** to module level:
    `MAX_TP_GROUP_SIZE` (`scheduler/mod.rs`),
    `MAX_MODEL_FILE_BYTES` (`huggingface/download.rs`),
    `MAX_DEVICE_NAME_BYTES` / `MAX_MODELS_HOSTED` /
    `MAX_MODEL_NAME_LEN` (`pool/manager/gossip.rs`),
    `MAX_INPUT_ITEMS_QUERY_LEN` / `INPUT_ITEMS_DEFAULT_PAGE_SIZE` /
    `INPUT_ITEMS_MAX_PAGE_SIZE` (`responses/mod.rs`).
  - **`check_signed_freshness`** added to `credit/ledger.rs`,
    `pub(crate) const CLOCK_SKEW_TOLERANCE_SECS` /
    `BALANCE_REPORT_MAX_AGE_SECS` exported. The
    dispatch-mod credit-transaction freshness check now uses the
    helper instead of duplicating both the constants and the
    one-sided staleness logic. Gotcha #82 records the helper as the
    canonical entry point for any new signed credit-typed message.
  - **Hardcoded `0.7` / `0.9`** in `responses/stream.rs` →
    `super::DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P`. Streaming-path
    fallback now matches the non-streaming path which already
    referenced the named constants.
  - **`compare.js` inline error extraction** → `U.extractErrorMessage`
    helper (with JSON-stringify fallback to preserve debug info).
  - **`dashboard.js`** — peer-label fallback `'unknown'` →
    `I18n.t('utils.unknown_model')`; removed dead `|| tier`
    fallback (`I18n.t` returns the key on miss, never falsy).

After R91 the tree has **887 lib tests + 75 integration tests**
passing, clippy clean on both feature sets, both default and
`--features llama` compile. Sweep log
(`.claude/sweep-log.jsonl`) totals **825 entries** across 91 rounds.

### Deferred-item follow-ups (R72/R75 leftovers, 2026-04-29)

Three commits cleaning up the structurally-deferred items the sweep
arc surfaced:

- **Worker crash-loop backoff** (`bab361c`) — `ModelProcessPool` now
  tracks per-model spawn failures; arriving requests during the
  cooldown window get `ServiceUnavailable` instead of waiting for
  `WORKER_CONNECT_TIMEOUT_SECS=30s` per attempt. Backoff steps
  1→2→4→8→16→32→60 s, reset on first successful spawn.
- **Cancellation token in inference loop** (`bab361c`) —
  `InferenceRequest.cancel: Option<Arc<AtomicBool>>` (`#[serde(skip)]`),
  `SharedState.cancel_signals` map keyed by an opaque token via
  the `x-swarmllm-cancel-token` HTTP header. Pipeline checks
  `request.is_cancelled()` per-token. Wired end-to-end through
  `responses/background.rs` so `/v1/responses/{id}/cancel`
  actually interrupts in-flight inference within one forward.
- **Stop-sequence KV truncate** (`9f9f22e`) —
  `pipeline/distributed.rs` sends a finalising
  `LayerForward(truncate_kv_to=ptc, activations=[])` to every remote
  segment after a stop string fires on a session-keyed request.
  Without this the next session turn would see the stop tokens
  still in the remote KV and produce contaminated output.
- **Escrow refund on inference failure** (`9f9f22e`) —
  `finalize_request` now calls `refund_escrow` on the `Err` arm.
  Previously `refund_escrow` had no production caller; failed
  requests left credits locked until the cleanup tick. Credit
  enforcement isn't gating users yet, but the bookkeeping is solid
  for when it is.
- **Pre-release surface trim** (`1d7eb39`) — every internal module
  in `src/lib.rs` now `#[doc(hidden)]`; `api`, `config`, `error`,
  `types`, `update` are the documented stable API. New
  `tests/integration/end_to_end.rs` exercises the full HTTP +
  shutdown lifecycle (`#[ignore]`'d, ~10 s).

C1 (binary signing) stays open with a fully-researched options write-
up at `memory/signing_options.md` — recommendation is **minisign**;
landing it requires a key-custody decision from the maintainer.

### Sweep arc R69 → R75 (2026-04-29 / 2026-04-30)

Long-running self-managed sweep covering: post-audit follow-ups (R69),
hot-path performance (R70), concurrency + lifecycle (R71), error
recovery + resilience (R72), observability + operability (R73), API +
wire-format correctness (R74), and pre-release readiness (R75). Each
round spawned 4–5 parallel review agents, applied auto-fixable
findings in batched commits, and surfaced architectural items for
follow-up. ~20 commits, broad coverage:

- **Performance** — `PendingLayerResultGuard` shared between dsd /
  speculative; `req_id_str` and `SamplingParams` no longer cloned per
  decode tick in `step_decode_pool`; per-token `default_eos` HashSet
  hoisted out of the decode loop; `verify_tokens.to_vec()` dropped
  from spec-round LayerForward; admin/stats and websocket
  build_stats_message no longer hold RwLock guards across blocking
  work; `health/monitor.rs` switched from `block_in_place` to
  `spawn_blocking` for the 30s sysinfo refresh; frontend
  notifications + dashboard render skips on equal state +
  sessionStorage debounced.
- **Concurrency** — `batch_scheduler_loop` and
  `pipeline_stream::spawn_accept_loop` + `handle_inbound_stream` now
  observe the watch-channel shutdown signal; PendingLayerResultGuard
  applied to speculative.rs's two leak sites; `apply_pipeline_guard`
  RAII catches panic-induced leaks of `active_pipelines` /
  `active_count`; `dashboard_rx` Lagged path now sends a re-sync
  message instead of silently dropping.
- **Resilience** — `apply_update` collapsed into a single signature
  that always re-checks version; HF probe wrapped with the
  NETWORK_RETRY_DELAYS exponential backoff (was a bare `.await?`);
  pool slaves now forward the `penalty_serve_failure` to the master
  via the same path as success-case spends (was hitting the slave's
  local balance and gating its own future requests); update.rs
  `info.downloaded = true` only set when staging path is the
  preferred (same-filesystem) location, not the temp_dir EPERM
  fallback.
- **Observability** — credit spend logs promoted from debug to info
  with `DIAG:` prefix (the earn side already had it; the spend side
  was dark); `escrow_held` + `escrow_pending_count` exposed on
  `/api/admin/credits`; `subnet_counts` retain on register so a NAT
  flip doesn't pile a peer into multiple buckets.
- **API correctness** — `SwarmError::InsufficientCredits` returns a
  distinct `insufficient_credits` error_type instead of the
  `rate_limit_error` mismatch (SDK retry logic was treating credit
  stops as rate-limits); `SwarmError::Network` maps to 502 +
  `network_error`; streaming finish_reason='error' replaced with
  'stop' (not a valid OpenAI value); `/v1/models` `created` is the
  manifest publish_date timestamp not Utc::now() (was unstable per
  call); `sse.rs::data_frame` emits a structured error on
  serialization failure instead of silently emitting an empty
  `data:` line; spec_logits doc-comment in swarmllm-types matches
  gotcha #29's γ+1 contract.
- **Hygiene** — `WorkerOptions` struct bundles run_worker's 7 runtime
  knobs (was 13 args under `#[allow(too_many_arguments)]`);
  `SplitLoadOptions` for split/loader; `speculative::softmax`
  wrapper deleted; `// SYNC:` comment dropped (gotcha #18 is the
  canonical 3-path warning); inline "deferred ChaCha encryption"
  paragraphs collapsed to one ARCHITECTURE.md pointer; mmap SAFETY
  comment expanded with candle's qtensor_from_ggml copy-semantics
  proof.
- **Release prep** — i18n files now sorted alphabetically by key for
  reliable parity audits (1015 keys + 2 metadata = 1017 entries per
  locale, all 21 languages confirmed in parity); `default.toml`
  documents the speculative-fast-path ChaCha bypass next to
  `enable_encryption = true`; `yamux_substream` CI tests now run on
  macOS (the Linux-only guard was a copy-paste from the
  multi-process integration tests).

890 lib tests pass throughout; clippy clean both feature sets;
pre-push hooks passing on every commit.

### Performance

- **`PrefixCache` lookup hit no longer needs a write lock** —
  `Entry::last_hit` is now an `AtomicU64` driven by a per-cache logical
  clock; the read-lock walk records the hit via a `Relaxed` store
  instead of upgrading to write. Removes the entire write-acquire from
  every cache hit and its sleep-based test fixture. Touches
  `src/inference/split/prefix_cache.rs` only.
- **`local_exec` holds `loaded_model_info` once per batch item** — was
  acquired twice per request (chat-template prompt + stop-string list).
  Now extracted from a single guard.
- **`KvCacheManager::check_multi_turn_reuse` takes `&HashSet<NodeId>`
  instead of `&[NodeId]`** — the per-holder `contains()` was O(peers).
  Caller in `inference/router/mod.rs` collects the DashMap directly
  into a HashSet so the conversion is paid once.
- **`GET /api/admin/responses` streams + bounded heap** — new
  `Database::for_each_json` helper streams JSON-encoded records without
  materialising a full Vec; the listing endpoint maintains a min-heap
  of size `≤ limit` ordered by `created_at`. Memory now O(limit)
  instead of O(total_records); only survivors get the full preview JSON
  built.

- **zstd compression on `WIRE_TAG_PREFIX_KV`** — flag-gated via
  `NetworkConfig::prefix_kv_compression` (default off). Send-side reuses
  the existing tensor-compression helpers and falls back to raw when the
  compressed form isn't smaller. Receivers always decompress regardless
  of the flag, so flipping it on a single peer doesn't require a
  coordinated upgrade. Expect 30–50% wire reduction on KV snapshots
  (zero-padded regions compress well); WAN bench will decide default-on.
- **Hot-path syscall savings** — gated per-layer `Instant::now()` in
  `split/executor.rs` behind `tracing::enabled!(TRACE)` (28 syscalls/token
  on a 28-layer model wasted at default log level), and same for
  `pipeline/distributed.rs` per-token `fwd_start` behind `enabled!(DEBUG)`
  with the matching DIAG emit dropped to debug-level for consistency.
- **Worker IPC capacity** — bumped reader→main channel from 16 to 64 in
  `model_worker.rs`. The admit-coalescing drain loop pulls 16/tick and a
  single decode tick can be 100–500 ms on CPU 7B; bursts would block the
  reader on `.send()` and delay the cross-node `PrefixFetchResult`
  fast-path short-circuit.

### Security & validation

- **Black-hat sweep 2026-04-29** — six parallel adversarial reviewers
  (network, crypto/auth, pool/credit, HTTP API, inference path, supply
  chain). 7 commits landed (`f4ff02b..fff42b4`); see
  `memory/audit_2026-04-29.md` for the full rollup. Highlights:
  - **Network**: gossip timestamp staleness in `events.rs` was
    symmetric `saturating_sub` — doubled the replay window to ~10 min;
    now one-sided per gotcha #44. PEX response no longer calls
    `kademlia.add_address` on unauthenticated peer/multiaddr pairs
    (Kademlia eclipse vector). `spec_logits` decoder hard-caps
    `num_positions ≤ 32` and `vocab_len ≤ 512_000`.
  - **Pool & credit**: `TREE_POOL_REMOVAL_REPLAYS` is no longer cleared
    on restart — the 5-min freshness window IS the replay window, and
    a saved `PoolRemoval` could re-evict after a planned restart. Now
    timestamps each entry. `handle_inbound_acceptance` verifies the
    acceptance's `invitee_node_id` matches the pending invitation
    (anyone learning the `invitation_id` could otherwise steal the
    slot). `track_forward_participation` caps peer-controlled
    token count at 8192 — `LayerForward.token_count = u32::MAX`
    would otherwise mint ~43B credits per serving node per flush.
  - **Inference**: intermediate-segment activations are now
    shape-validated against the input we forwarded; a malicious peer
    returning a wrong-shaped tensor would otherwise crash the next
    worker (gotcha #20). NaN/Inf rejected from peer-supplied f32
    tensors at three sites (ring AllReduce, TP attn/ffn AllReduce
    result, `spec_logits` rows before argmax in
    `greedy_accept_reject` — IEEE 754 NaN argmax is non-deterministic
    and lets a malicious peer steer accepted tokens).
  - **Shard & adapter integrity**: shard startup size check tightened
    to exact match (was ±10%); LoRA safetensors now BLAKE3-pinned via
    a `<filename>.blake3` sidecar so a swapped adapter can't silently
    produce wrong inference output.
  - **HTTP API**: `POST /api/admin/update/{check,apply}` now
    loopback-only (auto-downloads + binary swap should not be remote).
    `provider-model-status` rejects path-injection in `model_id` via
    allowlist (`[A-Za-z0-9._:@-]`, max 256, no `..`). `join-network`
    rejects private/loopback/link-local multiaddrs (P2P-layer SSRF).
    `GET /api/admin/responses` only returns `input_preview` /
    `output_text_preview` to loopback callers (single shared API key
    leaked prompt prefixes across users). Removed unreachable dead
    code: the non-loopback `x-swarm-internal-token` block in
    middleware.rs (per gotcha #30, `internal_auth_token` is
    per-process random and never crosses node boundaries).
  - **Update + supply chain**: `apply_update` re-verifies version is
    strictly newer than `env!("CARGO_PKG_VERSION")` at apply time
    (downgrade-by-replay protection). All GitHub Actions in `ci.yml`
    and `release.yml` pinned to commit SHAs with the tag in a trailing
    comment; `.github/dependabot.yml` opens weekly bumps. `libc::umask(0o177)`
    around the AF_UNIX socket bind in `process_pool.rs` so the IPC
    socket is created at 0o600 atomically (closes a TOCTOU between
    bind and post-bind `set_permissions` where a local attacker
    racing inotify on `/tmp` could connect first and impersonate
    the worker). `db.redb` chmod'd to 0o600 on Unix after create.
    `.env` loaded BEFORE the Tokio runtime spawns worker threads
    (`std::env::set_var` is unsound in a multi-threaded process).
  - **Deferred (still open)**: C1 — auto-update binary signing.
    SHA256 sidecar comes from the same release as the binary, so a
    compromised account/CI token publishes both together. Real fix
    needs an offline keypair embedded as `env!()` pubkey + cosign
    or minisign signature as a third release asset. Tracked in
    `docs/ARCHITECTURE.md` § Deferred Items.

- **PRIVACY: `chat_completions` no longer leaks prompt content into
  the activity event bus** — the event message included up to 60
  characters of the last user message via `prompt_preview`, and
  `emit_activity` broadcasts to every authenticated dashboard
  subscriber AND replays via `activity_history` to new connections.
  On a multi-tenant or shared-host node this leaked one user's
  prompt to others without their knowledge. The Anthropic handler
  did NOT reproduce this pattern. Removed the preview entirely;
  the event now carries only model, message count, and max tokens.
- **`lock_shard` / `unlock` now propagate DB write errors** — the
  handler used `let _ = db.insert_raw / .remove(...)` for the
  `locked_shards` persist, silently discarding write failures while
  still mutating the in-memory DashMap. On a DB error the handler
  returned `status: "ok"` despite a state divergence from disk;
  after restart the auto-manage pruner could remove a shard the
  operator believed was pinned. Same pattern was already corrected
  for pool pins, auto-manage policy, encrypted pipeline, and HF
  trust pin in earlier sweeps. Now persists first and surfaces any
  error as 500.

- **CRITICAL: `ResponsesRequest.extras` was an unbounded ingress vector**
  — the `#[serde(flatten)]` catch-all for unknown top-level JSON keys had
  no cap on count or per-value size. A request with thousands of unknown
  keys (or one very large value) was materialised into the in-process
  `HashMap<String, Value>` before any validation could reject it AND
  forwarded verbatim to upstream cloud providers on the proxy path.
  Added `MAX_RESPONSES_EXTRAS_COUNT = 32` and
  `MAX_RESPONSES_EXTRA_VALUE_BYTES = 4096`, checked first in
  `validate_responses_ingress`.
- **`/v1/responses/{id}/input_items` query parameters length-capped** —
  `after`, `before`, `order`, `include` were unbounded. Synthetic
  `item_N` cursors are short by construction so any large value is
  hostile. Capped at 64 bytes each at handler entry.
- **CSP meta tag in `frontend/index.html` synced to server header** —
  the meta lacked `blob:` in `img-src`, `frame-ancestors 'none'`,
  `base-uri 'self'`, and `form-action 'self'`. The server header in
  `middleware.rs::security_headers` already had these and takes
  precedence at runtime, but the meta covers the
  static-server / file-open / cached paths where the daemon's header
  isn't applied. Added an inline comment marking middleware.rs as
  authoritative.
- **CRITICAL: `responses.js` was calling a non-existent function** —
  `App.data.authFetch` doesn't exist; the symbol is `App.authFetch`.
  Every Responses dashboard interaction (load, retrieve, cancel,
  delete) was silently failing with `TypeError`. Same pattern that
  was caught in `auto-manage-status.js` previously — the responses.js
  refactor that introduced the `_action` helper reintroduced it.
  Fixed via 3-site replace.
- **`/v1/responses/{id}` path-param validation** — added
  `validate_response_id` helper mirroring the
  `previous_response_id` cap (≤64 ASCII alphanumeric `+_-`). Called
  at the top of `get_response`, `cancel_response`, `delete_response`,
  `list_input_items`, and `get_response_maybe_stream`. Closes a path
  where a megabyte-long `{id}` could inflate logs / DashMap key
  materialization for `BACKGROUND_CANCEL` / `BACKGROUND_STATE`.
- **`GET /api/admin/responses ?status=` length cap** — raw status
  string was split with no length cap, allowing an unbounded
  `Vec<String>` allocation. Added a 256-byte cap before splitting.
- **`Q8_0` tensor non-finite guard** — F32 path already rejected
  NaN/Inf in deserialized activations; Q8_0 path dequantized blindly.
  Added a matching `is_finite()` check after `dequantize_q8_0` so a
  malicious or broken peer can't poison subsequent attention via
  NaN/Inf-dequantizing Q8_0 blocks. Also flipped the F32+Q8_0
  truncation error type from `Internal` (HTTP 500) to `Inference`
  (a truncated wire payload from a peer is a network fault, not a
  local code bug).

- **`/v1/responses` ingress validation** — `validate_responses_ingress`
  runs BEFORE the cloud-proxy / Anthropic-bridge / local-inference
  branches. Caps `previous_response_id` ≤64, `instructions` ≤2 MB,
  `user` ≤256, `model` 1..=256, `truncation`/`service_tier` ≤64,
  `metadata` aggregate ≤64 KB. Closes a path where attacker-sized
  strings could reach upstream provider request bodies and log lines.
- **`/v1/messages` `max_tokens` cap** — Anthropic handler now rejects
  `max_tokens > 32768` (matches the local sampling-params clamp ceiling)
  at ingress instead of silently clamping for local + forwarding raw to
  upstream proxies.
- **HF-proxy rate limiting** — `/api/admin/hf/probe` and
  `/api/admin/hf/search` no longer get the loopback admin-GET exemption.
  A runaway local script or a malicious browser extension on
  `localhost:8800` can no longer loop-call them and burn HuggingFace
  API quota.
- **Pool invite-code validator** — now checks ASCII alphanumeric AND
  length (was: length only after trim+uppercase, accepting some malformed
  inputs silently).
- **Identity leaderboard `?limit=`** — clamped to `[1, MAX]` (was: silent
  empty array on `limit=0`).

### CI

- **macOS test + clippy** — `.github/workflows/ci.yml` matrix now runs
  `cargo test --lib --bins` and clippy on `macos-15` in addition to
  Linux. Integration tests stay Linux-only with explicit guards until
  the first macOS failure decides whether to fix or skip-list per test.
  Build job already had macOS — this closes the test-coverage gap from
  `docs/plans/next_steps.md` § 4.

### Correctness

- **`pool::handle_leave_pool` lock ordering** — was the only pool
  handler holding the `pool_state` read lock across
  `rate_limiter.check_and_record()`. Now matches
  `handle_create_invitation` / `handle_accept_invitation`: extract
  `pool_id` under the guard, drop, rate-limit, then later acquire the
  write lock for mutation. All 4 pool handlers audited for ordering
  consistency.

### Reliability

- **CRITICAL: M9 background spawn now `catch_unwind`s a panic** —
  the non-stream `/v1/responses` background path was a bare
  `tokio::spawn` with no panic guard, while the V8 streaming path
  has had `AssertUnwindSafe(...).catch_unwind()` around it since
  V8 landed. On a panic anywhere in `run_background_inference`
  (translate / chat_completions / buffer / parse / chat→responses
  translate) the redb record was stranded at `status:in_progress`
  forever — no terminal state ever written — and a polling client
  would see `in_progress` indefinitely. Wrapped the spawn body in
  `catch_unwind`; on panic we stamp a terminal `failed` record into
  redb and call `unregister_background_cancel`.
- **`POST /v1/responses` (M9 background) now returns 202 Accepted**
  — was returning HTTP 200 even though the response was queued for
  async completion. V8 streaming was already correct. Clients that
  branch on status code to detect sync vs deferred completion would
  have misclassified M9 background as a synchronous result.

- **`BACKGROUND_CANCEL` TTL sweep** — the response-id → cancel-flag
  registry could leak entries when a Tokio task was cancelled
  externally (e.g. process shutdown mid-flight) before its cleanup
  path ran. Added a parallel `BACKGROUND_CANCEL_AGES` map keyed by
  response id with insert-time `Instant`; new
  `register_background_cancel` / `unregister_background_cancel`
  helpers wrap every insert / remove site so the two maps stay
  consistent. New `prune_stale_background_state` drops entries
  older than `BACKGROUND_CANCEL_MAX_AGE_SECS = 7200` (2 h —
  generously above any real background-inference run) plus the
  matching `BACKGROUND_STATE` entry. Wired into the existing
  hourly responses sweep.
- **`/api/admin/network_code` peer-iter caps** — the dashboard's
  invite-code refresh handler walked every peer and every
  advertised address with no bound. Added
  `NETWORK_CODE_PEER_SCAN_CAP = 64` and
  `NETWORK_CODE_ADDR_PER_PEER_CAP = 16`. A public-facing IP is
  almost always advertised by the first few peers, so capping the
  inner loops preserves the happy path and bounds the worst case.

### Polish

- **`dispatch_inference` / `router_inference_stream` no longer
  thread an unused `&AppState`** — the parameter was passed
  through both functions and read by neither. Dropped from both
  signatures + 3 call sites in `openai/mod.rs`.
- **`is_prefill` DIAG field uses `PREFILL_ACTIVATION_THRESHOLD_BYTES`
  instead of a hardcoded `100_000`** — the constant is `pub(crate)`
  and documented as a tuning knob; the bare literal would silently
  diverge the diagnostic from the real classifier.
- **`responses.js` `_statusCell` fallback fixed** — `I18n.t(key,
  string)` passes the second arg as an interpolation **object**,
  not a fallback string. Switched to the
  `translated !== key ? translated : raw_status` pattern used
  elsewhere so unknown future status values render as the raw word
  rather than the literal i18n key.
- **`notifications.js` removed a leftover `console.warn`** — the
  ws-ticket fetch catch block logged on every transient reconnect
  failure (visible in DevTools console). The user-visible
  connection-lost banner via the `onclose` path is the right
  surface; logging adds nothing.

### Observability

- **`forward_batch` fallback log gained `model_id`** — the
  `kv_offset mismatch — falling back to sequential` debug log was
  missing the model key, making it impossible to correlate which
  model triggered the fallback in a multi-model deployment.

### Refactor / dedup

- **`buffer_and_translate_chat_response` helper** collapses the
  three-step buffer/parse/translate boilerplate that
  `create_response` and `run_background_inference` each hand-rolled
  with its own error-handling shape. Returns
  `Result<ResponsesResponse, String>`; each caller wraps the
  human-readable error string into its native error type. Site 1
  went from 13 lines + 2 match arms to 8 lines + 1 `.map_err`;
  site 2 went from 32 lines + 3 match arms to 14 lines + 1 match.
- **`speculative::greedy_accept_reject`** extracts the bit-identical
  14-line accept-reject arithmetic shared between
  `try_speculative_distributed` (Item 2) and `try_dsd_distributed`
  (Item 12). Token emission and gamma-controller bookkeeping that
  surrounds the call stay per-path because the divergence there is
  real.
- **`build_chat_completion_response` helper** in
  `api/openai/streaming.rs` collapses the response-shape
  construction shared between `router_inference` (with logprobs +
  session id) and `split_non_stream_response` (without). The
  empty-vec → `None` logprob conversion lives in one place.

- **`ResponseError::new(code, message)` constructor** replaces 11
  identical `ResponseError { code, message, extras: HashMap::new() }`
  struct literals across `responses/{mod, stream}.rs`.
- **`new_response_id()` / `new_message_id()` helpers** in
  `responses::mod` replace the
  `format!("resp_{}", uuid::Uuid::new_v4().simple())` /
  `format!("msg_{}", ...)` idiom that appeared at 7+ sites across
  the responses module. The `rs_` (reasoning item) and `resp_test_`
  (test fixture) prefixes intentionally stay inline — different
  conventions.
- **`apply_shard_window_change` helper** in
  `admin_models::helpers` extracts the post-window-compute /
  pre-event-emit block shared between `unload_shard` and
  `load_shard`: empty-window→`evict_and_unload` else
  `restart_with_window` + `evict_split_models`, then
  `clear_model_load_history` and `signal_dashboard(ModelsChanged)`.
  The activity-event + tracing tails stay per-handler (different
  messaging is the right answer there).
- **`DEFAULT_TEMPERATURE` / `DEFAULT_TOP_P` promoted to `pub(super)`**
  in `responses::mod`, mirroring the existing `DEFAULT_MAX_OUTPUT_TOKENS`
  shape. `translate.rs` now imports via `super::` instead of holding
  its own private literal — closes a drift hazard.
- **`_isActiveDlState(state)` helper** in `downloads.js` replaces 4
  duplicated `(state === 'downloading' || state === 'awaiting_manifest')`
  predicates and normalises a missing `typeof === 'string'` type guard
  in one of the four sites.
- **`detect_tp_groups` dropped its unused `_manifest: &ModelManifest`
  parameter** — the function operates on `candidates` and `segments`
  only.

- **SSE parser dedup across `responses/{stream,anthropic_bridge}.rs`** —
  promoted `drain_sse_data_payloads` and `find_subslice` from private
  to `pub(super)`, extracted a new `parse_sse_block_data_lines` helper
  for the per-block parsing both call sites need, and dropped
  `anthropic_bridge::find_event_boundary` (was the same logic under a
  different name).
- **`speculative_common_eligible` helper** — `speculative.rs` and
  `dsd.rs` shared a 6-line `eligible()` prefix (decoding flag, draft
  model loaded, greedy temperature, no encryption, etc.). Extracted
  to `pipeline/mod.rs`; each path now calls it after its own
  path-specific flag check.
- **`network-map.js` map-stats helper** — `render()` and
  `updateFromWs()` built the same I18n-formatted node/region count
  text inline; extracted `_updateMapStats(totalNodes, totalRegions, maxCount)`.
- **`responses.status_unknown` i18n key** — added across all 21 locale
  files; was missing, so any non-English user seeing a response with
  an unrecognized status value got the raw English `unknown`.
- **`auto-manage-status.js` plural fallback** — the active-download
  fallback string used `shard(s)` parenthetical pluralization;
  replaced with a proper singular/plural branch.
- **`detect_tp_groups` cleanup** — dropped an unused `_manifest:
  &ModelManifest` parameter that did nothing at the call sites; the
  function operates on `candidates` and `segments` only. Also renamed
  `_is_last` → `is_last` in `tensor_parallel.rs::execute_tp_segment`
  (the underscore convention falsely implied unused; the param is
  read at lines 316 and 357).

- **`CreditDelta` enum replaces `is_earning: bool`** on
  `apply_credit_direct`. Variants `Earning` / `Spending` / `Refund`;
  the new `Refund` leaves both monotonic counters
  (`lifetime_earned`/`lifetime_spent`) untouched, which is the
  semantics escrow refunds need but couldn't get from the old bool.
  `EscrowManager::create_escrow` and `refund_escrow` now route through
  the helper instead of hand-rolling the balance write + persist.
  `cleanup_expired` keeps its manual block (its retry-on-failure
  semantics need an in-memory revert that `apply_credit_direct`
  deliberately doesn't do); the reason is documented inline.
- **`pipeline::fastpath_request_disqualified`** helper extracts the
  shared TP-empty + LoRA + vision-images guard from
  `remote_generate.rs`, `speculative.rs`, and `dsd.rs`. Per-path
  shape / encryption / flag preconditions stay separate because they
  ARE subtly divergent (1-segment vs 2+-segment vs all-remote, and
  remote_generate has a per-model encryption gate the others don't).
- `SharedState::resolve_peer_id_bytes` helper replaces a 5-site
  `peer_id_map.or_else(peer_registry)` lookup duplication across
  `inference/pipeline/{distributed,remote_generate,speculative,dsd,
  tensor_parallel}.rs`.
- `crypto::hkdf_sha256_derive_32` helper replaces 3 sites that
  duplicated the same `Hkdf::new + expand + .expect("32 bytes is a
  valid HKDF-SHA256 output length")` pattern across `provider_keys`,
  `gossip_seal`, `session`.
- `cli::discover_model` helper replaces a 14-line `GET /v1/models`
  block duplicated in `bench.rs` + `chat.rs`.
- `auto_manage::manager::read_shard_pins` helper replaces a `pool_state.try_read()`
  + `shard_pins` extraction duplicated in `scoring.rs` + `prune.rs`.
- `cli::bench::tokens_per_sec` and a `pub const SWARMLLM_GITHUB_REPO`
  in `update.rs` (previously bare `"enapt/SwarmLLM"` string in two
  places).
- Various small stale-doc + dead-code cleanups; see commits `36af419`,
  `c9acbfc`, `c10956e`, `ccfbf14`, `712a4da`, `d8a840c` for the
  per-sweep summaries.

### Tests

- 821 lib tests passing (was 816 at v0.1.0). Clippy clean both feature
  sets. Pre-push hooks enforce `cargo fmt && cargo clippy --all-targets
  -- -D warnings` on every commit.

## v0.1.0 — 2026-04-25

First non-alpha tag. Cuts off the v0.1.0-alpha.2 line and rolls every
post-alpha commit (`v0.1.0-alpha.2..main`, ~56 commits) into a single
release.

### What's new since alpha.2

- **OpenAI Responses API (`/v1/responses`)** — full v1 (M1–M9) + v2
  (V1–V8) surface. Cloud-proxy passthrough, local inference via Chat
  translation, Claude/Anthropic-Messages translation bridge,
  multimodal input, redb persistence with 30-day TTL, chained
  `previous_response_id`, background mode, resumable SSE +
  background streaming (202 + Location handshake), `input_items`
  pagination, dashboard panel under the new Responses tab. See
  `docs/plans/responses_api.md` and `docs/plans/responses_api_v2.md`
  for design + full bench numbers in `docs/bench_results/`.
- **Audit fixes (`audit_2026-04-24.md`)** — 2 CVE backports for
  vendored libp2p-gossipsub (CVE-2026-33040 + CVE-2026-34219), 5
  cargo-update CVE patches (quinn-proto + rustls-webpki), spec_logits
  IPC wire-format fix (no more 5–7× JSON bloat for f32 row-major
  payloads), Anthropic / OpenAI proxy unknown-field preservation +
  beta-header forwarding, cache-token usage roundtrip, escrow refund
  loss + false-toast + total_bytes + spawn_region + no_peers_interval
  fixes, hot-path mem::take in forward_through_segments. See
  `audit_2026-04-24.md` in memory for full chronology.
- **Audit fixes (this session)** — DNS TOCTOU closed via custom
  reqwest resolver that filters private IPs at request time; shard
  size check tightened to exact-match for zero-hash placeholder
  manifests; prune cycle re-checks resource pressure between scan and
  execution so over-pruning can't happen; Grafana docker-compose
  refuses to start without explicit env vars (no more silent
  admin/admin).
- **Dep upgrades** — axum 0.7 → 0.8 (route patterns rewritten,
  WebSocket Message types adapted, async_trait removed, tower-http
  TimeoutLayer signature change), redb 3 → 4 (1.5× write throughput,
  on-disk format compatible via existing UpgradeRequired migration
  path), tower-http 0.5 → 0.6, candle 0.9 → 0.10 (vendor was already
  at 0.10.1 — refreshed dep declaration to match).
- **Distributed inference** — Items 1–7 + Item 8 (full architecture
  + cross-over demo: TinyLlama 1.1B GPU loopback ~100 ms slower vs
  control on small prompts; Qwen2.5-7B CPU loopback **12.9× iter-1
  TTFT** on 640-token prompt), Item 12 DSD multi-segment spec
  decoding (flag), Item 13 Q8_0 activation compression (~3.76× wire
  saving, flag), Item 16 Parallax scheduler phases A+B+B.2+C+C.2 on
  by default. Headlines: prefix-cache hit gives 29.4× wall-clock on
  repeat prompts; remote-generate fastpath gives 1.93× decode; Phase
  4 batched chunked prefill gives 1.57× tok/s @ c=4. See
  `docs/plans/distributed_inference_speedup.md`.
- **Cross-platform IPC** — Daemon ↔ model-worker IPC ported from
  Unix-socket-only to the `interprocess` crate (AF_UNIX with 0o600
  perms on Linux/macOS, named pipes with default-DACL on Windows).
  Runtime-validated on Windows (CPU + GPU, single-node + multi-node
  + split shards) on 2026-04-23. See note below.

### Test + lint baseline at v0.1.0

- 816 lib tests passing (782 → 816 since alpha.2), clippy clean both
  with default features and `--no-default-features --features
  dev,claude-subscription`.
- Integration tests green.
- 38/38 M1–M9 + 27/27 V1–V8 curl matrix pass against a live daemon.

### Known deferred items (post-v0.1.0)

- **libp2p 0.55 → 0.56** — ✅ landed post-v0.1.0 (commit `be9c32c`).
  Vendored libp2p-request-response re-ported to upstream 0.29.0 with the
  Tokio watchdog patch re-applied; the `confirmed`-flag patch (gotcha
  #16) is OBSOLETE in 0.29 and was dropped. Vendored libp2p-gossipsub
  removed entirely — upstream 0.49.4 ships the CVE-2026-33040 +
  CVE-2026-34219 fixes our backport carried.
- **`POST /v1/responses/compact`** — V9 of the v2 plan, deferred
  indefinitely until a concrete caller asks for it.
- See `docs/ARCHITECTURE.md` § "Deferred Items" for the full list.

### Notes for upgraders from v0.1.0-alpha.2

- Provider proxy now uses a custom DNS resolver. Hostnames that
  resolve to private/internal IPs (RFC 1918, link-local, loopback)
  are rejected at request time. The pre-flight `validate_provider_url`
  helper still runs for friendlier error messages but is no longer
  the only defense.
- `monitoring/docker-compose.yml` no longer supplies `admin/admin`
  defaults for Grafana. Run `cp .env.example .env && $EDITOR .env`
  before `docker compose up -d`.
- Anyone with a v3 redb file: the existing `Database::open` path
  already backs up the file (`db.redb.bak`) and recreates fresh, so
  in-place upgrade Just Works but loses prior state. New deployments
  start clean on v4.

---

### Detailed change log: alpha.2 → v0.1.0

> Prior tag: `v0.1.0-alpha.1` (2026-03-18, 674 tests) → `v0.1.0-alpha.2` →
> `v0.1.0` (2026-04-25, 816 tests). The summary above is the user-facing
> rollup; this section preserves the per-feature detail captured during
> the alpha → 1.0 grind so reviewers don't have to read 56 commits.

#### Cross-Platform IPC

Daemon ↔ model-worker IPC was Unix-socket-only between 2026-04-18 and
now, silently breaking Windows builds. Ported to the `interprocess`
crate (`local_socket` + tokio): AF_UNIX with 0o600 perms on
Linux/macOS, named pipes with default-DACL (current-logon-session)
on Windows. Security parity, no protocol changes (the framed codec
was already transport-agnostic).

**Runtime-validated on Windows (2026-04-23):** Both the CPU
(`swarmllm-windows-x86_64-cpu.zip`) and GPU
(`swarmllm-windows-x86_64-gpu.zip`) Windows binaries from this release
were smoke-tested on a real Windows host (Ryzen 7 5800H + RTX 3070
Laptop, 8 GB VRAM).

- **CPU single-node:** startup → BLAKE3 shard verify → named-pipe IPC
  handshake → worker subprocess model load → inference (HTTP 200 in
  5 s on TinyLlama 1.1B Q4_K_M) → API-triggered graceful shutdown
  (drain in 3 s, no orphaned worker).
- **CPU multi-node, single-segment:** node B bootstrapped from node A
  over loopback TCP. Noise + Yamux handshake green, GossipSub
  propagated model availability, node B's `/v1/models` listed
  TinyLlama as `owned_by: network`. Cross-node inference via Item 4
  remote-generate fast path: 8 tokens, trust score updated.
- **CPU multi-node, split shards (forced 2-segment pipeline):** A
  hosting shard 0 (layers 0-12), B hosting shard 1 (layers 12-22),
  auto-manage disabled to prevent cross-fill. Pipeline scheduler
  assembled `["A:0-12", "B:12-22"]`, encrypted activation tensor
  (172 KB) shipped over libp2p, B's worker subprocess computed the
  tail layers, response channel returned tokens. Generated coherent
  output ("The capital of France is Paris…") in 6 s.
- **GPU single-node:** dynamic-loading cudarc resolved bundled CUDA
  redist DLLs (cublas64_12.dll, cublasLt64_12.dll, cudart64_12.dll,
  curand64_10.dll, nvrtc64_120_0.dll, nvrtc-builtins64_124.dll) at
  process load — no CUDA Toolkit on the test host. RTX 3070 detected,
  worker subprocess loaded model on `device=Cuda(CudaDevice(1))`,
  inference returned coherent content. Graceful shutdown clean.

**Side-effect validation:** auto-manage's peer-to-peer shard transfer
also exercised on Windows during the multi-node run — A's missing
shard 1 was downloaded from B and vice versa via libp2p
request-response before auto-manage was disabled for the split test.

macOS aarch64 binary remains compile-validated only.

#### Distributed Inference Speedup Arc

A multi-session effort to speed up distributed inference, tracked in
`docs/plans/archive/distributed_inference_speedup.md`. Items 1–16 numbered in
plan order; default-on items landed as they shipped, flag-gated items
are off until benchmarked on real workloads.

**Default-on stack (user-facing in [Performance chapter](docs/book/src/operations/performance.md)):**

- **Item 3 — Continuous batching** (2026-04-19): fused `forward_batch`
  over concurrent Generate requests. 1.34–1.55× GPU throughput at batch
  2–8. CPU falls through to sequential with no regression.
- **Item 4 — Remote-generate fast path**: single-segment distributed
  inference runs the full decode loop on the remote worker instead of
  per-token coordinator round-trips. **1.93× decode speedup**.
- **Item 5 — Cross-request prefix cache**: worker keeps an LRU of prefill
  KV snapshots keyed by prompt prefix. **29.4× wall-clock** on
  re-submission of the same 513-token prompt.
- **Item 7 Phase 1+2 — BatchGenerate + Sarathi chunked prefill**
  (2026-04-19): SlotTable admits concurrent requests, each Prefilling
  slot advances by `prefill_chunk_tokens` (default 128) per decode tick.
  **17–23× TTFT fairness** at concurrency 2/4/8 on RTX 3070 +
  TinyLlama Q4. See `docs/plans/benchmarks/round4.md`.
- **Item 7 Phase 4 — Batched prefill forward** (2026-04-19): fuses
  concurrent same-shape prefill chunks into one `forward_batch`.
  **1.57× aggregate tok/s @ c=4** with uniform 180/180/180 ms TTFT.
  See `docs/plans/benchmarks/round5.md`.
- **Item 8 — Cross-node prefix KV sharing** (2026-04-19/20): when node B
  receives a prompt whose prefix peer A already prefilled, B fetches
  A's KV snapshot over the wire instead of re-prefilling locally.
  Full pipeline: PrefixCacheAnnounce gossip → cross-node index →
  PrefixFetchProbe → trust-gated SendPrefixKvFetch → BLAKE3 verify →
  NaN/Inf scan → hydrate → suffix-prefill.
  **Measured 12.9× iter-1 TTFT speedup on Qwen-7B CPU-CPU localhost**
  (151.7 s → 11.8 s on a 672-token prompt, Round 6 bench 2026-04-20).
  TinyLlama on GPU is the fast-prefill corner case where the fetch path
  is ~100 ms slower than re-prefilling (28 MB snapshot vs 460 ms
  prefill).
- **Item 16 — Parallax scheduler** (2026-04-18/19): shortest-path DP
  over observed per-layer latencies (EMA over recent forwards), replacing
  the greedy latency-only sort. Phase B.2 cross-gossips top-32 observed
  latencies via `NodeCapability.observed_latencies`. Phase C.2 adds a
  soft acquire/prune bias in `AutoShardManager` driven by a per-shard
  stability counter (≥3 consistent ticks before it acts); hard
  constraints (pinning, trust, VRAM) always win.

**Flag-gated:**

- **Item 2 — Distributed speculative decoding** (`speculative_distributed`):
  draft-target speculation across nodes. 40–52% accept rate in a
  llama-cpp-draft / candle-target pairing.
- **Item 6 — SWIFT self-speculative** (`swift_self_speculative`):
  target model acts as its own draft by skipping a layer range. Shelved
  on CPU until flash-attn-with-mask lands.
- **Item 12 — DSD (decentralized speculative decoding)**
  (`decentralized_spec_decoding`, 2026-04-18): multi-segment pipeline
  with γ-token speculation + KV truncation primitives + ~410 LOC
  coordinator loop in `pipeline/dsd.rs`. End-to-end WAN benchmark
  pending.
- **Item 13 — Activation compression Q8_0** (`activation_compression`):
  intermediate pipeline hidden states quantized to Q8_0 on the wire.
  ~3.76× compression, RMS error <0.005. End-to-end multi-segment
  benchmark pending.
- **Item 1 — Persistent pipeline stream** (`persistent_pipeline_stream`):
  one long-lived libp2p bidirectional stream per pipeline session.
  Wire-verified; no measured latency win because the bottleneck was
  elsewhere (Items 4 + 7 solved it).

#### Round 6 Bench Findings (2026-04-20)

The Item 8 two-daemon loopback bench caught three wire bugs before the
measured numbers above landed:

1. `SwarmMessage::PrefixCacheAnnounce` missing from the `TOPIC_MODELS`
   arm in `NetworkManager::handle_broadcast` — Phase 1 announces
   silently dropped at the gossip layer. Loopback self-index path
   masked it in single-node tests.
2. `WorkerMsg::PrefixSnapshotResponse` / `DaemonMsg::PrefixFetchResult`
   carried `payload: Option<Vec<u8>>` inside the JSON-framed IPC header.
   `serde_json` encodes `Vec<u8>` as a JSON array of integers (~5× size
   bloat), so a 28 MB snapshot became a ~102 MB header and blew past
   the 64 MiB `MAX_HEADER` cap.
3. Three chained cross-node-fetch timeouts (`PREFIX_FETCH_TIMEOUT_MS=500`
   in the worker, 400 ms daemon network timeout, 500 ms serving-worker
   IPC timeout) were sized for TinyLlama's 28 MB snapshot. A Qwen-7B
   snapshot is 73 MB and takes ~500–1000 ms to serialize+wire — every
   timeout fired and silently converted real hits into misses. Bumped
   to 3000 / 2500 / 2000 ms respectively, keeping the worker timeout as
   the outer bound.

#### Code Sweep (105 issues found, 58 fixed)
- **Round 1**: 10 parallel review agents across all 109 .rs files — 68 issues (9 CRITICAL, 32 HIGH, 22 MEDIUM), 41 fixed
- **Round 2**: Second pass — 37 new issues (5 CRITICAL, 22 HIGH, 10 MEDIUM), 17 fixed
- Key fixes: max_seq_len 2048 cap, ShardReader cross-tensor bleed, TensorPayload auth, escrow double-charge, IPC framing overflow, API key leak, Gemma embedding scale in forward_batch, hardcoded sampling in distributed forward

#### Credit System Overhaul
- Balanced rates: `rate × tokens` on both earn and spend (no layer multiplier)
- Minimum balance enforcement: `MIN_BALANCE_FOR_INFERENCE = -1000`
- Atomic credit accumulation via `pending_credit_earn` AtomicI64
- Anti-Sybil peer balance deduplication by NodeId
- Priority tiers require positive balance for Gold/Platinum

#### Device Pool Invite Codes
- 8-char one-time codes (e.g., `A3F7K2M9`), 24h expiry, Ed25519 signed
- CLI: `swarmllm pool create/invite-code/join/status/leave`
- API: `/api/pool/generate-code`, `/api/pool/join`, `/api/pool/device-name`, `/api/pool/credit-split`

#### Pool UX Overhaul
- Device nicknames, online/offline status, per-device stats, combined VRAM display
- QR code for invite codes, credit split configuration (0-50%)
- "My Devices" tab with full management UI

#### Terminology Clarification
- "My Devices" vs "Swarm Peers" — clear separation in setup wizard, share popover, dashboard

## [0.1.0-alpha.2] - 2026-03-18

### Release & Scale Readiness (Phase 19)
- **Docker release packaging**: Production `docker-compose.yml` (CPU default, GPU via `--profile gpu`), `.env.example` with all configurable env vars, GitHub Actions CI/CD pushing CPU + CUDA images to GHCR on git tag
- **Docker dev cluster**: 3-node `docker-compose.dev.yml` with static subnet, TCP bootstrap, container-optimized config (`config/docker-cluster.toml`)
- **Setup wizard redesign**: 4 steps → 3 steps (About You → Connect → Ready), invite code paste field, auto-download ON by default, hardware-aware model recommendations based on VRAM, dynamic summary
- **mDNS simultaneous-dial race fix**: When two nodes discover each other via mDNS simultaneously, both connections could fail. Added `pending_redial` queue with hash-based jitter (2-5s) for automatic recovery
- **Upload bandwidth enforcement**: `max_bandwidth_mbps` config now enforced on shard serving with proportional delay (was stored but never applied)
- **Manifest publisher claim**: Copied shard directories now properly gossiped — publisher set to local node_id with manifest hash recomputed on startup shard scan
- **Invite code error messages**: Invalid invite codes now return descriptive 400 errors instead of generic 500
- **Scalability (S1)**: Shard announce delta compression — only broadcasts when shard set changes or periodic re-announce every 10 cycles
- **Scalability (S2)**: P2P shard transfer fallback in auto-manage — when no HuggingFace source known, downloads from peer holders instead of doing nothing
- **Scalability (S3)**: peer_registry capped at 200 entries — evicts highest-latency non-LAN non-pipeline peers when over limit
- **Scalability (S4)**: Gossip broadcast frequency scales with `log(peer_count)` — 30s at ≤10 peers, 120s at 1K, 240s at 10K. Health pings stay at 30s
- **Docker image**: 181MB CPU image (debian:bookworm-slim), multi-stage build, non-root user, health checks
- **Tested**: 5-node Phi-3.5 distribution on Proxmox server (trust promotion, auto-manage, target replicas)

### Local Embedding Privacy
- **Config**: `local_embedding_privacy: true` in `[inference]` — requesting node embeds tokens locally, sends hidden-state activations (not raw token IDs) to remote first-segment nodes
- **LocalEmbedder**: Loaded from shard_000.bin at startup, uses candle for token→embedding conversion
- **Pipeline integration**: Pre-embedded activations skip remote embedding, reducing token exposure to relay nodes

### Deep Code Sweep — 56 fixes across 4 passes (16 parallel review agents)
- **Pass 1 (15 fixes)**: gossip_seal future-epoch removal, manager.rs unwrap safety + shard download cap, AllReduce zstd zip-bomb cap, shard atomic truncate, KV-cache eviction map fix, pipeline pending_vision cleanup + VLM hidden_dim expansion, escrow persist-before-balance, ledger bucket_balance div_euclid, error body truncation, manifest streaming BLAKE3, huggingface u64 range + progress retry, acquisition duplicate guard, protocol pre_embedded defaults
- **Pass 2 (10 fixes)**: sampling order consistency (temperature→top-k→softmax), TP block_in_place wrapper, stale logprobs clear, pending_tensor_channels leak fix (Instant timestamps + periodic sweep), num_layers saturating_sub, protocol unwrap→expect, anthropic proxy error truncation, escrow cleanup balance persist, model_id u16 length guard, max_tokens=0 early return
- **Pass 3 (17 fixes)**: duplicate streaming finish event guard, pending_layer_results failover leak, KV-cache orphan eviction, multi-turn session overwrite prevention, TP tp_size minimum guard (≥2), peer_http_url LAN/Tailscale fix, DB backup-on-upgrade, inverted --shards range validation, tied_output_weight streaming read, pool cosign cryptographic separation, pool_registry cleanup, escrow cleanup count, gossip epoch bound (reject >2 epochs old), invite code decrypt-fail no fallthrough, auto_update download gate, multi-image VLM remote guard, dead code removal
- **Pass 4 (14 fixes)**: completion_tokens EOS corruption (used clean_tokens.len → generated_tokens.len), keystore 0o600 permissions on private key files, WebSocket connection counter RAII guard, credit earn crash-window (single persist after earn+forwarding), MCP error body API key scrubbing, all_shards_available cache cap (1000 entries), TP attention GQA modulo wrap for tp_size > n_kv_head, AllReduce duplicate rank warning, pool state db.remove() on leave (fixes null deserialization type mismatch), health monitor future-timestamp evasion (clamp to zero), supervisor dead code cleanup (MAX_RESTART_ATTEMPTS), GGUF total_size saturating_add (overflow on malicious headers), gossip epoch fallback tightened (3→2 epoch window)
- **Model/storage (6 fixes)**: huggingface total_size==0 guard, retry HTTP status check, atomic tmp+rename for mmproj/header, LoRA rank==0 guard, auto_manage path traversal sanitization

### Feature Wiring — 8 previously unwired features now fully integrated
- **Priority tier enforcement**: `calculate_tier()` with real network percentile from peer credit gossip; `max_concurrent_for_tier()` enforces per-tier concurrent request limits in `drain_queue()`
- **Apply penalty on failure**: Credit penalty (configurable `penalty_serve_failure`, default -50) applied on distributed inference failure; penalty uses `apply_credit_direct` for immediate balance update
- **AllReduce registry cleanup**: `cleanup_stale()` removes entries where the receiver was dropped (timed out), wired into HealthMonitor's periodic 30s tick
- **Pipeline error broadcast**: `broadcast_pipeline_error()` notifies all pipeline participants on distributed inference failure, enabling peers to update shard availability
- **Pipeline affinity (KV cache reuse)**: Multi-turn sessions reuse previous pipeline assignment when all nodes are still connected, avoiding cold KV-cache on every turn
- **Relay service credits**: Tracks relay circuit open/close times in SharedState (`active_relay_circuits` DashMap), accumulates seconds in `relay_seconds_served` atomic counter, drains periodically in CreditLedger to `earn_relay_service()`
- **DHT record verification**: `verify_dht_value()` Ed25519 signature check on all Kademlia `GetRecordOk` results in NetworkManager — unsigned/invalid records are logged and ignored
- **Logprobs in API response**: `sample_token_with_params_and_logprobs()` in tensor_util collects per-token log probabilities via `SamplingContext`, stored in `PipelineExecutor.collected_logprobs` (Mutex), mapped to OpenAI-compatible `ChoiceLogProbs` in the `/v1/chat/completions` response. Works for split model (candle) inference paths

### Security Audit (Phase 16) — ~90 fixes across 5 rounds
- **Round 1-3**: Mandatory gossip signing, transport-authenticated dispatch, RFC 6479 anti-replay, signed DHT records, ephemeral key auth, path traversal fix, HF input validation, constant-time auth, CSP hardening, rate limiter cleanup, queue caps, input limits, WebSocket Origin validation, credit signature verification, XSS fixes
- **Round 4**: StreamingToken auth guard, peer IP bypass scoped to inference paths only, `.env` loader blocks dangerous env vars (LD_PRELOAD/PATH/DYLD_*), TOCTOU guard via `loading_models` DashMap with RAII `LoadGuard`, metadata hostname blocklist (Azure/AWS/DO/Oracle/Alibaba), IPv6 multiaddr extraction
- **Round 5**: All dispatch handlers require `authenticated_sender` (LayerResult, InferenceRequest, PipelineAssignment, InferenceError, TpAllReduceResponse), plaintext fallback removed (seal failure → drop), PEX SSRF filter (private/loopback/link-local IPs), shard serve requires peer_registry membership, pending_tensor_channels capped at 256, pending_tp_partials capped at 512, image_data 20MB cap, PoolMessage identity binding (CreditForward.from_node_id/MemberLeft.node_id must match sender), tool params size limits, lora_adapter validation, peer error body truncation/scrubbing, invite code capped at 4K, MCP research restricted to local/network models

### Frontend Polish
- CSS: removed unused variables, fixed hardcoded colors → CSS vars, removed duplicate rules, `@media (prefers-reduced-motion)`, light theme semantic color overrides
- Accessibility: `role="alert"` + `aria-live` on WebSocket banner, `aria-expanded` on hamburger, `aria-live="polite"` on chat messages, `scope="col"` on table headers
- JS: replaced inline styles with CSS classes, wired aria-expanded toggle

### Bug Fixes (Post-Audit)
- **Critical**: Credit balance overflow (`i64 +=` → `saturating_add`) and missing persistence in `track_forward_participation` — credits now survive daemon restart
- **High**: Divide-by-zero panic from malformed GGUF with `head_count == 0` (4 sites, remotely triggerable via HF probe)
- **High**: Silent null body sent to cloud provider on serialization failure (`unwrap_or_default` → proper error propagation)
- **Medium**: `model_request_counts` DashMap unbounded growth — now gated on registered models only
- **Medium**: `peer_shard_downloads` orphaned entries on peer disconnect — cleanup in ConnectionClosed handler
- **Medium**: Rate limiter cleanup task ignoring shutdown signal — now uses `tokio::select` with `shutdown_rx`

### Infrastructure
- **Workspace migration**: 3-crate Cargo workspace (`swarmllm`, `swarmllm-types`, `swarmllm-frontend`)
- **Ring AllReduce**: Bandwidth-optimal for ≥4 TP ranks, auto-selected by `choose_allreduce_strategy()`
- **Package distribution**: Homebrew formula, AUR PKGBUILD, deb/rpm packages, systemd service file
- **macOS CI**: Re-enabled on macos-15 runner
- **Docker**: Fixed Dockerfiles for workspace build
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### UX & Internationalization
- **i18n** — 20 languages (Arabic, Chinese, Czech, Dutch, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese)
- **Theme toggle** — Light / Dark / System theme with persistent preference
- **Basic/Advanced mode** — Toggle for simplified vs power-user UI
- **Plain-English UX pass** — Removed jargon, clearer labels and error messages for beginners
- **Compare UX** — Prompt textarea moved out of collapsed section, All/Local/Cloud filter buttons, chat source indicators, tok/s display fix for slow models (shows 0.5 instead of 0)
- **Provider UX** — `.env` file support for API keys, key source selector (auto/env/dashboard), error badges with click-to-settings
- **GPU OOM → CPU fallback** — Models that exceed GPU VRAM automatically retry on CPU (split fast-path preserved, not slow pipeline path)
- **Anthropic API model routing fix** — Requests now route to the correct model instead of always using the first loaded model

### Codebase Quality
- **Refactored**: `daemon.rs` (4015 lines → module directory), `admin.rs` (4225 lines → 4 modules), `split.rs` (10K lines → 6 modules)
- **Extracted**: `swarmllm-frontend` crate with dev mode for instant UI changes without full rebuild
- 674 tests passing (606 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Model Trust & On-Demand Loading (Phase 14)
- **Model Trust System** — demand-driven trust prevents trash models from auto-propagating
  - `ModelTrustLevel` enum: Discovered → Pinned → DemandVerified → NetworkPopular
  - Auto-manage only downloads shards for `DemandVerified`+ or user-`Pinned` models
  - Models promoted to `DemandVerified` after 3 real inference requests
  - `NetworkPopular` promotion when 3+ unique holder nodes serve a model
  - 7-day inactivity decay (Pinned models immune), persisted to redb
  - Trust level exposed in admin API (`trust_level` field on all model objects)
- **On-Demand Shard Loading** — inference requests trigger auto-loading from disk
  - Router detects shards on disk but not loaded in VRAM, triggers `check_and_load_model()`
  - LRU eviction makes room automatically (protected: active pipeline models)
  - Loading coordination via `DashMap<ModelId, Notify>` prevents concurrent loads
  - No more need to pre-load all models at startup
- **Kimi 2.5 support** — `k2*` prefix routing to Moonshot provider
  - Static fallback models: kimi-k2-0527, moonshot-v1-8k/32k/128k
  - Existing kimi* and moonshot-* routing preserved
- **UI improvements**
  - Trust level badges on model cards: Popular (green), Verified (accent), Pinned (yellow), Unverified (gray)
  - HF browser: prominent "On Swarm — N nodes" badge vs "New to network"
  - Download button renamed to "Add to node" (clarifies seed shard semantics)
  - "Local only" indicator when no peers host the model
- **Storage**: `get_all_json()` method on Database for key-value iteration with subkeys
### Claude Code Integration (Phase 13)
- **Full Anthropic Messages API** (`POST /v1/messages`) — complete Claude Code compatibility
  - `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking) request fields
  - `tool_use`, `tool_result`, `thinking`, `redacted_thinking` content blocks
  - `cache_control` on system blocks (Anthropic prompt caching)
  - Full pass-through to Anthropic cloud (all fields preserved including tools and thinking)
  - Anthropic→OpenAI translation proxy for non-Claude cloud models (GPT-4o, DeepSeek, etc.)
  - Tool calls and thinking blocks converted to text for local GGUF inference
  - `ResponseContentBlock` refactored from struct to enum (Text, ToolUse, Thinking variants)
- **MCP `compare` tool** — send same prompt to multiple models concurrently (up to 10)
  - Returns side-by-side results with `content`, `latency_ms`, `input_tokens`, `output_tokens`, `status`
  - Supports local, network, and cloud models in same comparison
  - Routes through `/v1/messages` for consistent routing logic
- **Claude Code as client**: `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`
- **Model Compare dashboard page** — side-by-side multi-model comparison UI with streaming
- 6 new unit tests (tool_use, tool_result, thinking, tools request, response serialization, internal conversion)
- 665 tests passing (597 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Published Benchmark Data
- **GPU (RTX 3070 8GB):** TinyLlama 1.1B 27.2 tok/s, Gemma-2 2B 20.6 tok/s, Phi-3.5 3.8B 46.4 tok/s, Qwen2.5 7B 29.0 tok/s
- **CPU (Ryzen 7 5800H):** TinyLlama 4.2 tok/s, Gemma-2 3.5 tok/s, Phi-3.5 1.8 tok/s, Qwen2.5 2.4 tok/s
- GPU speedups: 6.5x to 25.8x depending on architecture
- Methodology: 100 output tokens, 3-run average, single model loaded, Q4_K_M quantization

## [0.1.0-alpha.1] — 2026-03-07

First public release. Single Rust binary (~31MB) for decentralized P2P LLM inference.

### Inference Engine
- **11 model architectures**: Llama, Llama 4, Qwen2, Qwen 3.5 (hybrid SSM+attention), Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3 (MoE+MLA), GLM-4
- **4 architectures verified** with real models: Llama (TinyLlama-1.1B), Qwen2 (Qwen2.5-Coder-7B), Phi-3 (Phi-3.5-mini), Gemma2 (Gemma-2-2B-IT)
- **Distributed inference** verified on 2-node real LAN (WSL2 laptop + Proxmox server) with 5 models, crash recovery, auto-reconnect
- **Tensor parallelism** via AllReduce (star topology) with RTT-based LAN peer detection
- **VLM support**: LLaVA-v1.5-7B verified end-to-end (CLIP vision encoder + correct fine-tuned text model from second-state/Llava-v1.5-7B-GGUF), distributed mmproj, chat UI image upload (camera button, paste, drag-drop)
- **LoRA adapters**: per-request loading, verified with Qwen2.5-Coder-7B + rank-16 adapter
- **Speculative decoding** with draft model + rejection sampling
- **Cross-request batching** (GPU batch tensors, configurable `max_batch_size`)
- **Multi-turn KV-cache** with session reuse, cross-request prefix caching, chunked prefill
- **Flash attention** (CPU + GPU) and **paged attention** (CUDA block pool)
- **Structured output**: ResponseFormat API with JSON grammar state machine + schema validation
- **Sampling**: temperature, top-k, top-p, frequency/presence penalty, stop sequences

### API & Compatibility
- **OpenAI-compatible API**: `POST /v1/chat/completions` with streaming (SSE), `tool_calls`, `tool_choice`, `logprobs`, `top_logprobs`, Tool role
- **Anthropic Messages API**: `POST /v1/messages` — full Claude Code compatibility (tools, tool_choice, thinking, cache_control, metadata)
- **MCP server** at `/mcp` — `chat`, `models`, and `compare` (multi-model comparison) tools for Claude Code, Cursor, and MCP-compatible agents
- **12 cloud provider fallback**: OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi
- **Hidden states API**: `/v1/internal/hidden-states` for research (activation inspection, adapter insertion)
- **Embeddings**: `POST /v1/embeddings`
- **~62 admin REST routes** for dashboard, config, model management, downloads, providers
- **WebSocket** live updates (2s stats + prune event notifications)
- **Prometheus metrics** at `/metrics` (6 gauges + histogram)

### SDKs & Integrations
- **Python SDK**: `pip install swarmllm-client` — sync + async clients, streaming
- **JavaScript/TypeScript SDK**: zero-dependency, streaming support
- **LangChain integration**: `ChatSwarmLLM` provider
- **LlamaIndex integration**: `SwarmLLM` provider
- **Benchmark CLI**: `swarmllm bench` — sequential latency + concurrent throughput, JSON output

### Networking
- **P2P**: libp2p 0.55 with TCP+Yamux (primary) and QUIC transport
- **5-layer discovery**: mDNS (LAN), persistent peer cache (redb), encrypted invite codes, peer exchange (PEX), Kademlia DHT
- **NAT traversal**: libp2p relay circuits + DCUtR hole punching
- **GossipSub**: 6 topics for shard announcements, credit gossip, health, governance
- **Unified protocol**: `/swarmllm/1.0.0` — JSON control messages + binary tensor payloads (type-tag byte)
- **Wire compression**: zstd for tensor payloads

### Security
- **E2E encryption**: X25519 key exchange + ChaCha20-Poly1305 symmetric encryption
- **Forward secrecy**: ephemeral re-keying with key rotation
- **Sealed gossip**: all gossip messages authenticated (no plaintext fallback)
- **Replay protection**: nonce tracking + rejection
- **Shard integrity**: BLAKE3 content hash verified on every load
- **API auth**: Bearer token with auto-generation, loopback-only key retrieval
- **Provider key security**: at-rest encryption (AES-GCM), zeroize on drop, log scrubbing
- **Content-Security-Policy** header, IP-based rate limiting, CORS lockdown, SSRF protection
- **KV-cache privacy mode**: configurable per-session data isolation

### Model Management
- **Shard-only operation**: nodes download individual shards (~512MB each), never need a full model
- **HuggingFace integration**: search, browse, byte-range shard downloads with resume/retry
- **VRAM-aware auto shard management**: rarity-scored acquisition, popularity-based scoring
- **Smart shard pruning**: auto-remove over-replicated shards based on demand, resource pressure, and region diversity
- **Per-shard lock/pin** and per-model prune toggle
- **BLAKE3 integrity verification** on every shard load

### Credit System
- **Credit ledger**: earn credits by serving inference, hosting shards, seeding data
- **4 priority tiers**: Platinum (top 10%), Gold (top 30%), Silver (positive), Bronze (zero/negative)
- **Dual-signed transactions**: Ed25519 signatures from both parties
- **Credit escrow** for large requests
- **Anti-gaming**: rate limits, spot-check verification, subnet clustering detection
- **Sybil resistance**: trust scoring with decay, reputation tracking

### Identity & Pools
- **Ed25519 cryptographic identity** per node
- **Nicknames** with leaderboard
- **Device pools**: multi-device credit pooling with dual-signature invitation protocol

### Frontend
- **Embedded web dashboard** (vanilla HTML/CSS/JS, no build step, < 200KB)
- **4-step setup wizard** for first-run experience
- **Chat interface**: multi-turn + streaming, switchable Linear/Messenger layout, image upload (camera button, paste, drag-drop) for VLM models
- **Model browser**: HuggingFace search, shard grid visualization, download progress
- **Network map**: peer visualization with region grouping
- **Mobile-responsive** layout with dark theme
- **Reasoning model support**: DeepSeek R1 think token rendering

### Operations
- **Single binary**: ~31MB, zero runtime dependencies
- **CLI**: `run`, `status`, `chat`, `bench`, `peers`, `test-split`, `version`
- **Config priority**: CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Config hot-reload** via SIGHUP or API
- **Graceful shutdown**: SIGTERM handler with subsystem drain
- **Auto-updater**: checks GitHub Releases, downloads + self-replaces with restart prompt
- **JoinSet task supervisor**: automatic restart-on-crash for all 10 subsystems
- **Database**: redb v3 (embedded, ACID, ~15% faster than v2)

### Platform Support
- Linux x86_64 (CPU + CUDA + ROCm)
- macOS aarch64 Apple Silicon (Metal)
- macOS x86_64 Intel (CPU)
- Windows x86_64 (CPU + CUDA)

### Model Loading
- **Auto-extract gguf_header.bin** from shard_000.bin when header is missing (daemon pre-pass)
- **Single-GGUF manifest**: Full GGUF files stored as shard_000.bin generate 1-shard manifests (not split into logical shards)
- **Single-shard mmap fallback**: Models with 1 shard and no tensor entries load via mmap instead of ShardReader
- **Probed flag fix**: Models with gguf_header.bin correctly show as probed in admin API

### Test Suite
- 659 tests: 591 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E
- All passing, clippy clean, rustfmt clean
- CI: GitHub Actions (fmt → clippy → test → build)

[0.1.0-alpha.1]: https://github.com/enapt/SwarmLLM/releases/tag/v0.1.0-alpha.1
