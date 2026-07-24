# SwarmLLM Networking Plan — Reliable P2P Inference Across NAT (once and for all)

Status: **IN PROGRESS (2026-07-24)**. Owner: maintainer. Supersedes ad-hoc NAT
fixes in R143–R150.

## Implementation status

- ✅ **Phase 1 — application-level inference relay** (`RelayedEnvelope`,
  end-to-end sealed, single-hop, rate-limited; learned reverse routes; relay
  fallback in the directed-send path). Plus the load-bearing refinement:
  **prefer the app-relay over a flaky relay *circuit*** (per-peer direct-vs-circuit
  connection tracking) — this is what makes the both-NAT'd case actually work,
  since two circuit-connected peers otherwise looked "connected" and used the
  fragile circuit RR. Crypto in `crypto/relay_seal.rs`; transport in
  `network/manager/relay.rs`; state in `daemon/state/relay.rs`.
- ✅ **Tensor relay — distributed inference across NAT.** Extends the relay to
  the *distributed-pipeline* tensor path (`LayerForward`/`LayerResult`), so a
  multi-shard inference between two un-connectable nodes runs over the sealed
  app-relay instead of the flaky libp2p circuit. A new `SwarmRequest::RelayedTensor`
  + `WIRE_TAG_RELAYED_TENSOR` binary frame (tensors are large + already-encoded,
  so out of JSON), negotiated via `features::TENSOR_RELAY`. The tensor is
  **ephemeral-sealed for the target's STATIC key** (`crypto::relay_seal::
  seal_relayed_tensor`), NOT the session — sidestepping the "decrypt by transport
  sender" problem that made naive relaying of session-sealed tensors impossible.
  Forward and result each travel as a SEPARATE relayed request (never the RR
  response substream), which avoids the bidirectional-over-relay problem; the
  coordinator's existing `pending_layer_results` oneshot resolves on the
  relayed-back `LayerResult`. Send fallback wired into `handle_send_tensor` +
  `send_tensor_result_as_request`; the relay-unwrap stamps the origin's peer
  bytes so the result routes home.
- ✅ **Cross-cutting — additive protocol evolution.** Nodes advertise a
  `PROTOCOL_VERSION` epoch + a `features` bitfield (`swarmllm_types::features`);
  a sender gates a new/optional message type on the recipient's advertised bit,
  so older nodes stay on the common subset instead of failing to decode. Rule
  documented in `.claude/rules/architecture.md`. (Relay is `features::RELAY`.)
- ✅ **Phase 2 — prefer direct (implicit).** The directed-send path already
  uses a direct connection whenever one exists (`has_direct_connection`), and
  learned relay routes TTL-expire, so new work migrates off the relay onto a
  direct link as soon as one forms. Explicit *mid-stream* migration is a
  non-goal (a generation is short-lived; ~1 extra hop is cheap).
- ✅ **Phase 3 — scale + incentive.** (a) **Multi-relay routing**: nodes
  advertise the relays they're connected to (`relay_reservations`); a sender
  picks a relay the target can also reach and **fails over** across relays if one
  drops, so traffic isn't funnelled through a single anchor. Relay discovery
  rides existing `NodeCapability` gossip (`relay_capable` propagates), so a node
  learns every relay-capable peer without a new channel. (b) **Credit-metered
  relaying**: `relay_inference_bytes` earns at the shard-seeding rate, so
  donating relay capacity counts as a contribution — informational/priority
  only, since credits are not enforced for correctness. (c) **Mixed-version
  wire-compat test** (`node.rs::version_compat_tests`): an older peer's
  capability parses as feature-less, a newer peer's (with unknown fields) still
  parses — a network-breaking change now fails the build.

- ✅ **DHT relay discovery** (`network/discovery.rs::{start_providing_relay_service,
  query_relay_providers}`, `relay_service_key`): a relay-forwarding node registers
  under a well-known Kademlia key; a node short on relay connections
  (`MIN_RELAY_CONNECTIONS`) queries that key each discovery tick and dials the
  relays it finds — so the relay role survives losing the bootstrap anchor and
  decentralizes as the swarm grows its own public relays.
- ✅ **Mixed-version integration test** (`tests/integration/test_relay_mixed_version.rs`):
  the full A → relay → B round-trip at the wire + crypto level — the relay can't
  open the payload, the target recovers it exactly across two hops, a tampered
  header is rejected, and a feature-less (older) peer is gated out. Deterministic
  (no libp2p timing), so it's a reliable CI guard.

The **entire plan is implemented and tested** — Phases 1–3, the version
handshake, DHT relay discovery, and the mixed-version guard. Nothing remains
deferred. (A heavier full-daemon two-node libp2p test would add little over the
deterministic wire+crypto one and risks flakiness; not pursued.)

---


## 1. Why this matters (the adoption problem)

The target user is a home user behind NAT — Docker, WSL2, Windows, Linux, often
CGNAT or symmetric NAT. For that user the swarm must **just work**: join, be
reachable, serve and consume inference. Today it doesn't reliably: two NAT'd
nodes connect and gossip, but **P2P inference between them does not complete**.
Combined with **version-breaking network changes** (a node on vN can't talk to
vN±1), this is the single biggest adoption blocker. This plan fixes both.

## 2. Root cause (audited 2026-07-24)

Observed live (our WSL/NAT node ↔ an external tester on a public IP, both on
v0.3.16, directly connected):

- **Bidirectional request/response fails.** `tester→us` RR works (we log
  `ResponseSent`); `us→tester` RR (rr_ping health-checks AND `remote-generate`)
  gets **no response** → "peer never acknowledged". Inference needs the
  us→peer direction for BOTH the request and the token stream
  (`SendStreamingToken`), so it fails either way.
- **We depend on DCUtR (hole punching) for the data path.** DCUtR is enabled
  (`network/behaviour.rs`), but never fires/succeeds here — and per research it
  **cannot** succeed for symmetric NAT / CGNAT, which is common in 2026.
- **The relay is connectivity-only.** libp2p circuit-relay-v2
  (`network/relay.rs`) has `max_circuit_bytes = 1 GB`, `max_circuit_duration =
  1 h`, and is designed as a *brief bootstrap* before DCUtR upgrades to direct.
  It carries the connection + GossipSub, but the bidirectional
  `request_response` inference path over it is unreliable (reverse-substream +
  the "prefer a fresh direct dial that then can't round-trip" behaviour).

**In one line:** our architecture assumes the relay is temporary and DCUtR gives
us the data path. For the *majority* home case (NAT'd both ends, no
hole-punch), there is **no reliable data path at all**.

## 3. State of the art (research 2026-07-24)

- **Tailscale / DERP** (the reliability gold standard): every connection
  *begins* over a relay (a dumb ciphertext pipe over TCP/443), then **upgrades
  to direct opportunistically**. If the upgrade fails, the relay keeps carrying
  **all** traffic. ~95 %+ success under CGNAT, <2 s. The relay is a first-class
  **data path**, not just a signalling channel.
  ([tailscale.com](https://tailscale.com/blog/nat-traversal-improvements-pt-1),
  [sitepoint](https://www.sitepoint.com/tailscale-peer-relays-nat-traversal-derp/))
- **libp2p hole-punching / DCUtR** is real but probabilistic (~70 % on
  cone NATs, ~0 % on symmetric/CGNAT); the docs themselves position the relay as
  the necessary fallback.
  ([libp2p.io/docs/hole-punching](https://libp2p.io/docs/hole-punching/))
- **QUIC NAT-traversal draft** coordinates hole-punch attempts; still
  probabilistic, still needs a middleman when it fails.
  ([seemann.io](https://seemann.io/posts/2024-10-26---p2p-quic/))

**Takeaway:** hole punching is an *optimisation*, never a guarantee. The
guarantee comes from a relay that reliably carries the data. **We already have
the exposed middle-man — the anchor. We just aren't using it as a data path.**

## 4. The plan

### Guiding principle
> The relay (anchor) is a **reliable inference data path**, always available.
> Direct connections are an **opportunistic optimisation** layered on top.
> Correctness never depends on hole punching succeeding.

### Phase 0 — Pin the exact failure (1–2 days, no code risk)
Reproduce with two NAT'd nodes we control (not the external tester) OR get the
tester's logs. Answer: does our `us→peer` RR **arrive** at the peer and get no
reply (peer-side handler), or **never arrive** (relay/transport)? This decides
how much of Phase 1 is "fix the existing relay path" vs "add an app relay".
Instrument both sides via `docs/DIAGNOSTICS.md` (`DIAG:` on inbound RR receipt).

### Phase 1 — **Any reachable peer as an application-level inference relay** (the definitive fix)
The relay role is NOT tied to a dedicated anchor — it is any node that is
directly reachable (public IP, port-forward, permissive NAT, VPS, or opt-in
relay mode). The anchor is simply the **first** such node (the bootstrap); the
design must never hard-depend on it. When two peers have no working direct path,
route the inference **through any reachable peer** at the **application layer**,
independent of libp2p circuit-relay quirks:

- New `SwarmMessage::RelayedInference { target: NodeId, inner: sealed bytes }`.
  A (NAT'd) sends it to the anchor; the anchor forwards `inner` to B; B's
  `SendStreamingToken`s go back A the same way (anchor → A). The anchor is a
  **dumb pipe** — `inner` stays sealed with the existing per-hop ChaCha20
  (`crypto/pipeline_seal.rs`), so the relay **never sees plaintext** (matches the
  Layer-1 encryption invariant).
- The anchor already holds live connections to both peers (it's the
  bootstrap/relay), so forwarding is a `send_request` it can already make in the
  direction that works. This sidesteps the fragile relayed-RR-reverse-substream
  entirely.
- Selection: the router prefers direct → then relayed-through-anchor → then
  fail. `inference/scheduler` already ranks candidates; add a
  "reachable-via-relay" tier so a NAT'd holder is still usable.
- Anchor mode (`--anchor`) opts into inference *forwarding* (not serving) — a
  small, bounded, encrypted byte-forwarder. Rate/size-capped per peer
  (reuse the credit-forward rate-limit pattern).

This makes inference **complete for any pair that can both reach the anchor** —
i.e. the entire swarm, since every node already bootstraps to it.

### Phase 2 — Opportunistic direct upgrade (offload the relay)
Keep DCUtR, but treat it as an optimisation: once a direct path is confirmed
(AutoNAT-v2 `ExternalAddrConfirmed` on both ends, or a successful hole-punch),
migrate the inference stream off the anchor onto the direct connection. Never
block on it. Prefer dialling a peer's **public** address directly before falling
back to the relay (Phase 0 will confirm whether our current code already does
this and merely mis-round-trips).

### Phase 3 — Relay capacity, discovery, cross-platform consistency
- **Multiple relays / relay discovery**: today there is one anchor
  (`swarmllm.duckdns.org`). Publish relay candidates on a DHT topic so the swarm
  scales past one middle-man and survives its downtime. Any public node can opt
  in as a relay.
- **Capacity**: raise/verify `max_circuit_bytes` / `max_circuits` for the
  forwarding path; meter relay bytes into the existing credit system so relaying
  is a *contribution* (incentive-aligned, like serving).
- **One reachability abstraction across platforms** so behaviour is identical on
  Docker / WSL / Windows / Linux:
  - Docker: `--network host` guidance + the `172.17.0.1` filter (shipped R151).
  - WSL2: mirrored-mode auto-detect (shipped); UPnP multicast doesn't traverse
    WSL, so document the manual port-forward + `external_addresses` path, and
    make relay-fallback the default so WSL "just works" without it.
  - Native Windows/Linux: UPnP + AutoNAT-v2 + relay, same code path.
  - **Every platform gets the anchor relay as the floor**, so none of the above
    is *required* for basic function — only for the direct-connection speedup.

### Long-term viability — does this survive with NO dedicated anchors?

Yes, provided the relay role is decentralized (above). The reasoning:

- **Every P2P-over-NAT network requires a reachable subset** (IPFS public DHT
  nodes, BitTorrent seeds, Bitcoin DNS seeds). A 100%-strict-NAT swarm with zero
  reachable peers cannot self-organise — no rendezvous, no relay, no Kademlia.
  This is a property of NAT, not of this design. So "no anchor AND no reachable
  peer" is impossible for *any* architecture, not just ours.
- **The reachable supply is self-sustaining** via three mechanisms, all in this
  plan: (1) **discovery** — reachable peers publish as relay candidates on a DHT
  topic, found dynamically; (2) **incentive** — relay bytes metered into credits,
  so being a public relay *earns* like serving does, making public capacity
  economically self-supplying; (3) **hole-punch offload** — DCUtR migrates pairs
  to direct links, so relay demand scales *sub-linearly* (one lightweight relay
  serves many pairs; cf. a handful of DERPs for millions of Tailscale nodes).
- **The anchor is the seed, not the structure.** It bootstraps the very first
  swarm; as the swarm grows past a few dozen nodes, reachable members provide the
  relay mesh and Kademlia provides rendezvous. Losing the anchor then degrades to
  "slower to bootstrap a brand-new node", not "swarm stops working".
- **Healthy floor, not hard dependency:** a few community-run public relays
  (like IPFS public gateways) are good hygiene as a safety net, but nothing in
  the protocol may *require* a specific one.

### Cross-cutting — stop the version-breaking network churn (adoption)
This is half the adoption problem. Make network evolution **additive and
backward-compatible**:
- **Protocol-version handshake in `identify`**: nodes advertise a
  `swarm-net/<major>.<minor>`; a receiver negotiates the highest common minor.
  New message types are *optional extensions*, never a hard requirement.
- **Never repurpose or remove a `SwarmMessage` variant** across a release; add a
  new variant and keep handling the old one for ≥2 minor versions (the
  backup-artifact + AutoNAT-v1→v2 rounds show why: a peer on an older build must
  still interoperate).
- **Feature-negotiated transports**: QUIC / TCP / relayed all coexist; a node
  picks the best the *pair* supports, never a hard cutover.
- CI: a "mixed-version swarm" integration test (vN ↔ vN-1) so a network-breaking
  change fails the build.

## 5. Recommended sequencing

1. **Phase 0** now (diagnosis; cheap, unblocks the rest).
2. **Phase 1** next — the anchor inference relay is the "once and for all" fix
   and the highest leverage: it makes the *primary* use case (two home users)
   work regardless of NAT type, using the exposed middle-man we already run.
3. **Cross-cutting version-handshake** in parallel with Phase 1 (small, stops
   future adoption bleed).
4. Phases 2–3 as follow-ups (speed + scale, not correctness).

## 6. Explicitly NOT doing
- Tailscale / external overlay dependency (can't put the whole swarm on it;
  we build the DERP-style relay into our own anchor).
- Making correctness depend on hole punching.
- Any version-breaking wire change without a negotiated fallback.
