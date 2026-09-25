//! Node capability, peer info, and node-level stats.

use serde::{Deserialize, Serialize};

use crate::ids::{NodeId, ShardId};
use crate::pool::ContributionMode;

/// Wire-protocol epoch. Bumped ONLY on a genuinely breaking change to the
/// `SwarmMessage` wire format (a variant repurposed or removed — which the
/// project rule forbids without a negotiated fallback). Additive changes (a new
/// variant behind a `features` bit) do NOT bump this. A receiver that sees a
/// higher epoch than it knows treats the peer as "newer, some features I don't
/// speak" and keeps interoperating on the common subset, rather than failing.
pub const PROTOCOL_VERSION: u16 = 1;

/// Optional, additively-negotiated protocol features. A node advertises the set
/// it implements in `NodeCapability::features`; a sender gates an optional
/// message type on the recipient advertising the corresponding bit, so an older
/// node is never handed a variant it can't decode. This is the mechanism that
/// makes network evolution backward-compatible — new features are extensions,
/// never a hard cutover (see `.claude/rules/architecture.md`).
pub mod features {
    /// Understands the NETWORKING_PLAN Phase 1 `RelayedEnvelope` (can receive an
    /// inference message routed through a relay).
    pub const RELAY: u64 = 1 << 0;

    /// Understands the NETWORKING_PLAN tensor relay (`SwarmRequest::RelayedTensor`
    /// / `WIRE_TAG_RELAYED_TENSOR`): a distributed-pipeline tensor forward or
    /// result routed through a relay, ephemeral-sealed for the recipient's static
    /// key. A sender only relay-wraps tensors for a peer that advertises this bit.
    pub const TENSOR_RELAY: u64 = 1 << 1;

    /// Understands `LayerForward.next_hop`: after computing its segment, this
    /// node can forward the activations straight to the NEXT segment holder
    /// rather than returning them to the coordinator.
    ///
    /// A coordinator only sets `next_hop` for a peer advertising this bit, so an
    /// older node is simply never given one and keeps replying to the
    /// coordinator — which is the existing behaviour and always correct, just
    /// one round trip more expensive.
    pub const PIPELINE_CHAIN: u64 = 1 << 2;
    /// Direct peer chaining, second wire form: the chain trailer (0x06) is
    /// followed by a reply-to trailer (0x07) naming the COORDINATOR, so the
    /// tail of a chain answers the node that asked rather than the hop that
    /// handed it the activations. The first form never carried that identity —
    /// the receiver treated the forward's sender as the requester, which is
    /// right for one hop and wrong for a chain — so a v1-only peer would
    /// answer its predecessor and the run would time out. The planner
    /// therefore requires THIS bit; v1 is kept defined, never re-used.
    pub const PIPELINE_CHAIN_V2: u64 = 1 << 3;
    /// The node acknowledges a tensor forward the moment it is received and
    /// sends the computed result back as its own request, instead of holding
    /// the request open until the result is ready. A coordinator that sees this
    /// bit may treat "no acknowledgement within the ACK deadline" as a dead path
    /// and fail over in seconds; without it, a forward that lands on a peer that
    /// never answers costs the whole segment deadline (observed 2026-08-21:
    /// 300 s, twice in one request). Older coordinators still work: they accept
    /// an ACK response and a result arriving as a request.
    pub const FORWARD_ACK: u64 = 1 << 4;
    /// The node keeps the tokens of a reply it streams on the remote-generate
    /// fast path for a short while after sending them, and answers a
    /// `SwarmMessage::ResendTokens` naming a range by sending those tokens
    /// again. Each token of such a reply is its own fire-and-forget
    /// `request_response` send, so one lost send used to strand every token
    /// after it (gotcha #438); a requester that sees a hole in the sequence
    /// asks this peer to fill it instead of waiting out a deadline for
    /// nothing. A requester only asks a peer advertising this bit, and only
    /// answers a dropped inbound token with `SwarmResponse::Dropped` to a
    /// peer advertising it; an older peer keeps today's behaviour in both
    /// directions.
    pub const RESEND_TOKENS: u64 = 1 << 5;

    /// The node maintains and publishes a Vivaldi coordinate
    /// (`NodeCapability::coord`), so readers can estimate the round trip
    /// between it and any OTHER node carrying one.
    ///
    /// Unlike the bits above, nothing is ever *sent* on the strength of this
    /// one — it advertises a fact, not a message type, and the field is an
    /// `Option` that a reader already has to handle. It exists so a node can
    /// tell "this peer has no coordinate yet" from "this peer's build has
    /// none", which decides whether a missing coordinate is worth waiting for.
    pub const NETWORK_COORDS: u64 = 1 << 6;

    /// The node CONFIRMS a re-keyed session before the other end starts using
    /// it: having completed an ephemeral exchange it initiated, it immediately
    /// sends a `SwarmMessage::SessionKeyConfirm` sealed under the new key.
    ///
    /// A responder that sees this bit may therefore keep sealing with the key
    /// it already has until that confirmation opens — which is what stops a
    /// lost exchange reply leaving one end holding a key the other has never
    /// seen. Without the bit the responder adopts the new key immediately, as
    /// every build before this one did, because an older peer will never send
    /// a confirmation and waiting for one would strand the link instead.
    pub const SESSION_KEY_CONFIRM: u64 = 1 << 7;

    /// Understands the decoded-so-far trailer (`0x08`) on a `LayerForward`: the
    /// tokens generated so far, which the SAMPLING segment needs to apply the
    /// caller's `frequency_penalty` / `presence_penalty`.
    ///
    /// The field existed from the beginning and never reached the wire — both
    /// binary encoders omitted it and both decoders set it empty — so a
    /// distributed request whose last segment was remote had its penalties
    /// silently dropped. A coordinator only emits the trailer to a peer
    /// advertising this bit, because an older peer does not merely ignore an
    /// unknown trailer: it rebuilds the seal's AAD from the trailers it parsed,
    /// so an unrecognised one makes every encrypted forward fail to open.
    ///
    /// ⚠ The history alone did not make penalties work remotely: the penalty
    /// VALUES arrive only with `FORWARD_SAMPLING`'s trailer (2026-09-25).
    pub const FORWARD_GENERATED_IDS: u64 = 1 << 8;

    /// Understands the pre-embedded trailer (`0x09`) on a `LayerForward`: the
    /// payload is already embedded hidden states rather than prompt text.
    ///
    /// `LayerForward.pre_embedded` travelled ONLY inside the tensor-parallel
    /// trailer (`0x02`), which an ordinary pipeline forward never carries, so a
    /// node receiving a locally-embedded prompt read the flag as false and
    /// tokenised a float tensor as UTF-8 text. That is the whole of
    /// `inference.local_embedding_privacy`, whose entire purpose is to hand a
    /// REMOTE first segment hidden states instead of raw token ids.
    ///
    /// Gated for the same reason as the trailer beside it: an older peer
    /// rebuilds the seal's AAD from the trailers it parsed. A coordinator that
    /// cannot send this to the peer holding segment 0 refuses the request
    /// rather than silently sending tokens the caller asked to keep private.
    pub const FORWARD_PRE_EMBEDDED: u64 = 1 << 9;

    /// Understands the refusal trailer (`0x06`) on a `LayerResult`: a typed
    /// reason a forward was turned away before any of it ran
    /// (`ForwardRefusal`). A coordinator with this bit sends a forward its
    /// peer could not decrypt AGAIN, to the same peer, once the repair
    /// handshake the refusal armed has re-keyed the link — instead of failing
    /// over, which with one holder meant failing the request.
    ///
    /// An older coordinator would skip the trailer harmlessly (a result's seal
    /// does not depend on its trailers, unlike a forward's), but the serving
    /// node still only sends it to a peer advertising this bit, so the
    /// additive-protocol rule holds without an argument about each decoder.
    pub const FORWARD_REFUSAL_REASON: u64 = 1 << 10;

    /// Understands the sampling trailer (`0x0A`) on a `LayerForward`: the
    /// caller's temperature, top-p, top-k, penalties and logprobs, for the
    /// segment that turns logits into a token.
    ///
    /// `LayerForward.sampling` was in-process only, so a REMOTE last segment
    /// sampled with the worker's defaults (0.7 / 0.9 / 40, no penalties)
    /// whatever the caller asked: a request for temperature 0 came back
    /// sampled, differently each run, whenever the model was split and its
    /// last part was on another computer — the usual shape of a split
    /// (measured 2026-09-25, `docs/FUTURE_WORK.md` #106). Gated like the
    /// trailers before it: an older peer rebuilds the seal's AAD from the
    /// trailers it parsed, so one it does not know fails every encrypted
    /// forward.
    pub const FORWARD_SAMPLING: u64 = 1 << 11;

    /// The full feature set THIS build implements. Advertised by every node.
    pub const ALL: u64 = RELAY
        | TENSOR_RELAY
        | PIPELINE_CHAIN
        | PIPELINE_CHAIN_V2
        | FORWARD_ACK
        | RESEND_TOKENS
        | NETWORK_COORDS
        | SESSION_KEY_CONFIRM
        | FORWARD_GENERATED_IDS
        | FORWARD_PRE_EMBEDDED
        | FORWARD_REFUSAL_REASON
        | FORWARD_SAMPLING;

    /// Does `advertised` include every bit in `needed`?
    pub fn supports(advertised: u64, needed: u64) -> bool {
        advertised & needed == needed
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub gpu: Option<GpuInfo>,
    /// What this node's processor is, for a node that has no graphics card —
    /// and for one that does, since the processor still runs anything the card
    /// cannot hold.
    ///
    /// A graphics card has been described in full since the beginning
    /// ([`GpuInfo`]), while the processor had no representation at all, so a
    /// peer without a card appeared in the dashboard as the bare word "CPU".
    /// Every such machine looked identical to every other, from a fanless
    /// mini-PC to a sixteen-core server.
    ///
    /// Carries the same kind of information the card already does, and no more.
    /// Deliberately NOT the fingerprinting material the `os` field declines to
    /// send: a processor model is the hardware doing the work, which is exactly
    /// what a peer needs to judge, whereas an operating-system build number
    /// identifies the install rather than the machine.
    ///
    /// `#[serde(default)]` → `None` from a node predating the field, which
    /// renders as it always did rather than being guessed at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<CpuInfo>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    /// How much system memory this node will actually let a model take right
    /// now — its own admission ceiling, not the operating system's free-memory
    /// reading.
    ///
    /// The two differ by a lot and always in the same direction. A node sizes
    /// a swap-safe budget from its total RAM and its contribution level, so an
    /// 8 GB machine at the default level admits about 4 GB; `ram_available_mb`
    /// is `sysinfo`'s raw figure with no margin at all. A scheduler routing on
    /// the raw number offers segments that the receiving node's own gate then
    /// refuses — measured, twice against one peer inside two seconds, at 46
    /// then 28 layers, each costing a round trip before the refusal was known
    /// (report #022).
    ///
    /// `None` from a node predating the field, and `None` on a node whose
    /// models go to a graphics card (`vram_available_mb` answers for those).
    /// Unknown never excludes: a reader falls back to `ram_available_mb` and
    /// behaves exactly as it did before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram_model_budget_mb: Option<u64>,
    pub disk_available_mb: u64,
    pub bandwidth_mbps: f32,
    pub hosted_shards: Vec<ShardId>,
    pub max_contribution: ContributionLevel,
    pub uptime_seconds: u64,
    pub version: String,
    /// Voluntary ISO 3166-1 alpha-2 country code (e.g. "US", "DE").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// This node's Vivaldi network coordinate, so ANY reader can estimate the
    /// round trip between this node and another — including two peers neither
    /// of which is the reader.
    ///
    /// That estimate is the one fact the routing cost model never had: it
    /// prices each candidate by the reader's OWN round trip to it, so a chain
    /// of peers in one city and a chain spanning three continents came out the
    /// same. `region` is the coarse stand-in (same / adjacent / distant) and
    /// stays as the fallback.
    ///
    /// `None` from a node predating the field, and `None` on a node whose
    /// coordinate has not settled yet (`NetworkCoord::is_usable`) — a reader
    /// falls back to exactly what it did before, so unknown never excludes.
    /// Gated by [`features::NETWORK_COORDS`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coord: Option<crate::netcoord::NetworkCoord>,
    /// Estimated tokens/s for a 7B Q4 model based on GPU memory bandwidth.
    /// Used by the scheduler as a speed tie-breaker.
    #[serde(default)]
    pub est_tokens_per_sec_7b: f32,
    /// Operating-system family this node runs on — `linux` | `windows` |
    /// `macos` | other `std::env::consts::OS` value.
    ///
    /// Display and filtering only; nothing routes on it. Deliberately the OS
    /// *family* and not a version or build string: the leaderboard wants "what
    /// kind of machines make up this network", and a precise OS build is
    /// fingerprinting material that would be gossiped to every peer forever
    /// for no functional gain.
    ///
    /// `#[serde(default)]` → `None` from a node predating the field, which
    /// renders as "unknown" rather than being guessed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub os: Option<String>,
    /// Snapshot of the sender's observed per-layer latency EMA for other
    /// peers. Lets newly-joining nodes bootstrap Parallax routing from
    /// gossiped foreign observations instead of waiting for their own
    /// direct samples. Receivers merge each entry with a trust-weighted
    /// discount so low-trust senders can't poison routing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub observed_latencies: Vec<LatencyObservation>,
    /// NETWORKING_PLAN Phase 1 — this node will forward inference messages
    /// between two peers that cannot reach each other directly (it is
    /// publicly reachable and opted into relaying, e.g. an `--anchor`). Peers
    /// gate relay use on this flag before wrapping traffic for it, so the
    /// capability is negotiated, never assumed. `#[serde(default)]` (false)
    /// means an older node advertising no flag is simply never used as a relay.
    #[serde(default)]
    pub relay_capable: bool,
    /// Wire-protocol epoch this node speaks (see [`PROTOCOL_VERSION`]).
    /// `#[serde(default)]` (0) marks a pre-negotiation node.
    #[serde(default)]
    pub protocol_version: u16,
    /// Bitfield of optional protocol features this node implements (see
    /// [`features`]). A sender gates an optional/new message type on the
    /// recipient advertising the matching bit, so evolution stays additive and
    /// an older node is never handed a variant it can't decode. `0` (default)
    /// means "advertises no optional features" — the safe pre-negotiation base.
    #[serde(default)]
    pub features: u64,
    /// NETWORKING_PLAN Phase 3 — the relay-capable peers this node is currently
    /// connected to (bounded). Lets a sender that can't reach this node directly
    /// pick a relay the node is ALSO connected to, so the forward actually
    /// lands — the mechanism that makes multiple relays work rather than all
    /// traffic funnelling through one anchor. Empty (default) → the sender falls
    /// back to any relay it is connected to.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relay_reservations: Vec<NodeId>,
    /// This node runs in `--anchor` mode: a dedicated bootstrap/relay node that
    /// never serves inference and hosts no shards.
    ///
    /// Advertised so the dashboard can label it, because an anchor is otherwise
    /// indistinguishable from an ordinary peer that happens to hold nothing —
    /// which is exactly what a struggling node looks like. Deliberately NOT
    /// inferred from `relay_capable && hosted_shards.is_empty()`: an ordinary
    /// node that has relaying on and has not acquired a shard yet matches that
    /// too, and mislabelling it "anchor" would be worse than no label.
    ///
    /// `#[serde(default)]` (false) so a node on an older build simply carries
    /// no label, per the additive-evolution rule.
    #[serde(default)]
    pub anchor_mode: bool,

    /// Can this node run inference work **right now**?
    ///
    /// `false` means the node has detected that it cannot execute a request at
    /// all, however much memory or how many shards it advertises — so peers
    /// should route inference elsewhere. It keeps serving SHARDS regardless:
    /// a byte-range read needs no worker, and a node in this state is still the
    /// most useful thing it can be.
    ///
    /// Two conditions set it, and both are outages that report themselves as
    /// health (see `SharedState::inference_outage`):
    ///
    /// - the graphics stack died under a running node, so no worker of any kind
    ///   starts — this binary links `libcuda`, so a processor-only worker fails
    ///   in the loader exactly as a card-bound one does;
    /// - the message dispatcher has stopped consuming, so nothing inbound is
    ///   reaching this node in the first place.
    ///
    /// **`#[serde(default)]` must answer `true`, not `false`.** A node on an
    /// older build advertises nothing here, and reading that silence as "cannot
    /// serve" would route around every peer that has not upgraded — the exact
    /// failure `resident_layers`' third state exists to avoid. Unknown means
    /// "no reason to think otherwise", as it does for every other field here.
    #[serde(default = "serving_inference_unless_told_otherwise")]
    pub can_serve_inference: bool,

    /// Layers of each model this node currently has LOADED, by model id.
    ///
    /// `hosted_shards` says what is on disk; this says what is in memory right
    /// now and therefore already paid for. The distinction decides how a peer's
    /// spare capacity is priced: a node already serving a model has paid for the
    /// weights it is holding, and charging it for them again routes around the
    /// one machine best placed to answer (gotcha #329). Exempting EVERY layer
    /// instead, which is what a bare "is it warm" flag forces, credits it with
    /// weights it has not paid for.
    ///
    /// Petals announces the same thing for the same reason — each server
    /// publishes the contiguous blocks it is actively serving, not the blocks it
    /// could load — and clients route over those announcements
    /// (<https://arxiv.org/pdf/2209.01188>).
    ///
    /// `#[serde(default)]` (empty) per the additive-evolution rule. **Empty is
    /// ambiguous on purpose**: it means "this node said nothing about residency",
    /// which is true both of an older build and of a node with nothing loaded.
    /// A consumer must therefore treat a model's ABSENCE as no information and
    /// fall back to whatever it did before, never as proof the model is cold —
    /// reading absence as "not resident" would route around every peer running
    /// an older build, which is the failure the additive rule exists to prevent.
    #[serde(default)]
    pub resident_layers: Vec<ResidentModelLayers>,
}

/// The serde default for [`NodeCapability::can_serve_inference`].
///
/// Named rather than `|| true` so the reason survives: a node that says nothing
/// is not a node that has said no. See the field's own documentation.
fn serving_inference_unless_told_otherwise() -> bool {
    true
}

/// How many layers of one model a node has resident. See
/// [`NodeCapability::resident_layers`].
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ResidentModelLayers {
    pub model_id: String,
    pub layers: u32,
    /// The exact ranges behind `layers`, as this node's loader keys them.
    ///
    /// **The count alone cannot say which plans are free** (FUTURE_WORK #99).
    /// A worker keys its layers by exact `(start, end)`: an equal range loads
    /// nothing, a range that strictly contains held ones drops them before
    /// loading, and one contained in a held range is loaded a second time. So a
    /// node holding `[2..14)` credited with "12 resident layers" was planned
    /// `[10..22)`, needed 12 new layers, refused, and failed over. Petals routes
    /// on exactly this — each server announces the contiguous block span it
    /// serves, and clients route over the spans (<https://arxiv.org/pdf/2209.01188>).
    ///
    /// `#[serde(default)]` (empty), and empty keeps the count-only pricing an
    /// older build's announcement gets — the additive-evolution rule. Adding a
    /// field needs no feature bit here: gossip is JSON, and nothing in these
    /// types denies unknown fields, so an older node reads past it.
    #[serde(default)]
    pub ranges: Vec<ResidentLayerRange>,
}

/// One range a node's worker holds, and what loading a range that strictly
/// contains it would give back. See [`ResidentModelLayers::ranges`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ResidentLayerRange {
    pub start: u32,
    pub end: u32,
    /// Layers of memory the loader releases when a strictly containing range
    /// replaces this one — ZERO for the range a worker was spawned with, which
    /// its admission recorded at no cost (`process_pool::HeldRange`). A peer
    /// that sent nothing here is read as releasing nothing: the conservative
    /// side, since over-crediting it is what plans a load it refuses.
    #[serde(default)]
    pub releasable_layers: u32,
}

impl NodeCapability {
    /// How much memory this node can give a model's layers right now, on
    /// whichever device it would load them on.
    ///
    /// The single answer, because two callers were choosing the device
    /// themselves — the routing bound (`scheduler::max_hostable_layers`) and
    /// the capacity planner — and a third would have had to get the same
    /// two-line match right again.
    ///
    /// **A stated system-memory budget is how a node says its models load
    /// there**, and it is therefore tested before the card. That figure is
    /// computed only on the branch where models go to system memory, so its
    /// presence carries the placement decision and not merely a number.
    ///
    /// Which matters for a node that HAS a card and has been told not to use
    /// it (`inference.gpu_layers = 0`): it still gossips its `gpu`, because the
    /// card is really there, so keying on that field alone judged it by memory
    /// its models would never occupy while it loaded every one of them into
    /// RAM. Asking about the card first was right only while the card was the
    /// only thing that answered.
    ///
    /// Otherwise a graphics card answers with its free memory, which already
    /// excludes whatever is resident on it — the property warm-peer pricing
    /// depends on, and one the RAM budget shares because it nets off what is
    /// already committed. A node that has told us neither falls back to the
    /// operating system's free-memory reading, exactly as before this field
    /// existed: that reading is not a promise the node can keep, and routing
    /// on it produced segments the receiving node refused on arrival.
    pub fn memory_for_model_layers_mb(&self) -> u64 {
        match self.ram_model_budget_mb {
            Some(mb) => mb,
            None => match &self.gpu {
                Some(g) => g.vram_available_mb,
                None => self.ram_available_mb,
            },
        }
    }
}

/// One entry in `NodeCapability::observed_latencies`: the sender observed
/// this `peer` takes `ms_per_layer` to serve a distributed-inference
/// segment (averaged via the sender's local EMA).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LatencyObservation {
    pub peer: NodeId,
    pub ms_per_layer: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuInfo {
    /// Model as the operating system reports it, e.g.
    /// "AMD Ryzen 7 5800H with Radeon Graphics".
    pub name: String,
    /// Logical processors visible to the machine.
    pub cores: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
    /// Memory bandwidth in GB/s, looked up from GPU name.
    #[serde(default)]
    pub memory_bandwidth_gbps: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,
    Moderate,
    Maximum,
}

impl From<ContributionMode> for ContributionLevel {
    fn from(mode: ContributionMode) -> Self {
        match mode {
            ContributionMode::Minimal => Self::Minimal,
            ContributionMode::Moderate => Self::Moderate,
            ContributionMode::Maximum => Self::Maximum,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeStats {
    pub peers_connected: u32,
    pub requests_made: u64,
    pub uptime_start: chrono::DateTime<chrono::Utc>,
    /// NAT status detected by AutoNAT ("Public", "Private", "Unknown").
    #[serde(default)]
    pub nat_status: Option<String>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            peers_connected: 0,
            requests_made: 0,
            uptime_start: chrono::Utc::now(),
            nat_status: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub addresses: Vec<String>,
    pub capability: Option<NodeCapability>,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub latency_ms: Option<u32>,
    pub trust_score: f32,
    /// Raw libp2p PeerId bytes for directed request_response messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_id_bytes: Option<Vec<u8>>,
    /// Smoothed send-to-acknowledgement latency of tensor forwards to this
    /// peer, in ms — the RFC 6298 `srtt` the ACK deadline is built from.
    /// Measured on real work, so it sees queueing on a loaded peer that a
    /// ping cannot (gotcha #386). Routing prefers it to `latency_ms` when
    /// present. Local only, never gossiped: it describes OUR path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ack_srtt_ms: Option<u32>,
    /// Measured throughput to this peer in bytes per second — a windowed
    /// MAXIMUM over completed tensor forwards, not an average.
    ///
    /// The companion to `ack_srtt_ms` and deliberately the other half of the
    /// same split: the round-trip figure is taken only from SMALL forwards
    /// (where the time is the peer's) and this one is driven by LARGE ones
    /// (where the time is the payload's). A sample is dominated by one or the
    /// other and cannot measure both.
    ///
    /// It exists because loss on a healthy TCP path is absorbed by
    /// retransmission and so appears as a slower transfer, never as a failed
    /// forward — which is why the delivery-ratio term cannot see it, and why a
    /// peer at 60 ms with 3% loss out-sorted one at 81 ms with none while being
    /// 2.9x slower on a 513 KB payload (issue #21). It also captures a rate
    /// limit, which no small-message probe can detect.
    ///
    /// Local only, never gossiped: like `ack_srtt_ms` it describes OUR path to
    /// that peer, which is not a property of the peer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub goodput_bytes_per_sec: Option<u64>,
    /// How many forwards have moved the goodput filter. Carried beside the
    /// estimate so "nothing is measuring" is distinguishable from "measured,
    /// and the path is fine" — the ambiguity that hid #495 being inert.
    #[serde(default)]
    pub goodput_samples: u32,
    /// Active inference request count reported by this peer's last health ping/pong.
    #[serde(default)]
    pub active_request_count: u32,
    /// When this peer was first discovered (Unix timestamp).
    /// Used for leaderboard eligibility: peers must be at least `min_lifetime_days` old.
    #[serde(default)]
    pub first_seen: u64,
    /// Number of verified dual-signed credit transactions from this peer.
    /// Used for leaderboard eligibility: peers need `min_verified_transactions`.
    #[serde(default)]
    pub verified_transaction_count: u32,
    /// Whether this peer was discovered via mDNS (on the same LAN).
    /// LAN peers have ~1ms latency and are automatically preferred by the scheduler.
    #[serde(default)]
    pub is_lan_peer: bool,
}

/// Peer Exchange response — a list of known peer multiaddrs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeerExchangeResponse {
    /// Up to 20 known peer multiaddrs.
    pub peers: Vec<String>,
}

/// Mixed-version wire-compatibility guard (NETWORKING_PLAN cross-cutting). These
/// tests fail the build if a change makes a node on one release unable to
/// interoperate with a node on an adjacent release — the exact adoption blocker
/// the additive-protocol rule exists to prevent. Both directions of skew:
///  - an OLDER peer's capability (missing the new fields) must still parse, and
///    default to "advertises no features" so we never route it new traffic;
///  - a NEWER peer's capability (with fields we don't know) must still parse,
///    so we keep interoperating on the common subset.
#[cfg(test)]
mod version_compat_tests {
    use super::*;

    pub(super) fn base_fields() -> serde_json::Value {
        serde_json::json!({
            "node_id": vec![0u8; 32],
            "gpu": null,
            "ram_total_mb": 8192u64,
            "ram_available_mb": 4096u64,
            "disk_available_mb": 100000u64,
            "bandwidth_mbps": 100.0f32,
            "hosted_shards": [],
            "max_contribution": "Moderate",
            "uptime_seconds": 100u64,
            "version": "0.3.16-alpha"
        })
    }

    /// A node that predates `ram_model_budget_mb` states no budget, and a
    /// reader must fall back to the raw free-memory figure — behaving exactly
    /// as it did before the field existed. Unknown never excludes.
    #[test]
    fn a_peer_that_states_no_memory_budget_is_read_as_it_always_was() {
        let cap: NodeCapability = serde_json::from_value(base_fields()).unwrap();
        assert!(cap.ram_model_budget_mb.is_none());
        assert_eq!(
            cap.memory_for_model_layers_mb(),
            4096,
            "with no budget stated, the raw figure is all there is"
        );

        // And when one IS stated it is what routing weighs, because it is the
        // only one of the two the node will actually honour.
        let mut v = base_fields();
        v.as_object_mut()
            .unwrap()
            .insert("ram_model_budget_mb".into(), serde_json::json!(2048u64));
        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert_eq!(cap.memory_for_model_layers_mb(), 2048);

        // A card answers when the node has stated no system-memory budget —
        // which is what a node running its models ON the card reports. Its
        // free figure already excludes what is resident there, the property
        // warm-peer pricing depends on.
        let mut v = base_fields();
        v.as_object_mut().unwrap().insert(
            "gpu".into(),
            serde_json::json!({
                "name": "card",
                "vram_total_mb": 8192u64,
                "vram_available_mb": 6000u64,
                "memory_bandwidth_gbps": 0.0f32,
            }),
        );
        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert!(cap.ram_model_budget_mb.is_none());
        assert_eq!(cap.memory_for_model_layers_mb(), 6000);
    }

    /// A node that HAS a card and has been told not to use it
    /// (`inference.gpu_layers = 0`) still gossips that card, because it is
    /// really there. Its models nonetheless load into system memory, so
    /// judging it by video memory measured what its models would never
    /// occupy — and a peer scheduling onto it sized the segment from the
    /// wrong pool entirely.
    ///
    /// Stating the system-memory budget is how such a node says where its
    /// models go, which is why that field is tested before the card.
    #[test]
    fn a_card_the_node_will_not_use_does_not_answer_for_its_memory() {
        let mut v = base_fields();
        let obj = v.as_object_mut().unwrap();
        obj.insert("ram_model_budget_mb".into(), serde_json::json!(2048u64));
        obj.insert(
            "gpu".into(),
            serde_json::json!({
                "name": "card it will not use",
                "vram_total_mb": 8192u64,
                "vram_available_mb": 6000u64,
                "memory_bandwidth_gbps": 0.0f32,
            }),
        );
        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert_eq!(
            cap.memory_for_model_layers_mb(),
            2048,
            "the models load in RAM, so the RAM budget is what bounds them"
        );
    }

    /// The other direction: OUR announcement reaching a node that predates the
    /// field. Reproduced against a struct with the pre-change shape, because
    /// "serde ignores unknown fields" is cheap to check and expensive to be
    /// wrong about — a mixed-version swarm is the normal state during a
    /// rollout, and a deserialisation failure here would be silent.
    #[test]
    fn our_announcement_still_decodes_on_a_node_that_predates_the_budget_field() {
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct OldCapability {
            node_id: NodeId,
            ram_total_mb: u64,
            ram_available_mb: u64,
            disk_available_mb: u64,
            version: String,
        }

        let mut v = base_fields();
        v.as_object_mut()
            .unwrap()
            .insert("ram_model_budget_mb".into(), serde_json::json!(2048u64));
        let json = serde_json::to_string(&v).unwrap();
        let old: OldCapability =
            serde_json::from_str(&json).expect("an older node must still decode our announcement");
        assert_eq!(
            old.ram_available_mb, 4096,
            "and reads the field it does know, unchanged"
        );
    }

    #[test]
    fn old_capability_parses_as_featureless() {
        // A pre-negotiation node announces NONE of the new fields.
        let cap: NodeCapability = serde_json::from_value(base_fields()).unwrap();
        assert!(!cap.relay_capable);
        assert_eq!(cap.features, 0);
        assert_eq!(cap.protocol_version, 0);
        assert!(cap.relay_reservations.is_empty());
        // The negotiation gate correctly refuses to send it relay traffic.
        assert!(!features::supports(cap.features, features::RELAY));
    }

    #[test]
    fn future_capability_with_unknown_fields_still_parses() {
        // A NEWER node announces extra fields we don't know about, and feature
        // bits beyond the ones we implement.
        let mut v = base_fields();
        let obj = v.as_object_mut().unwrap();
        obj.insert("version".into(), serde_json::json!("9.9.9"));
        obj.insert("relay_capable".into(), serde_json::json!(true));
        obj.insert("protocol_version".into(), serde_json::json!(9u16));
        // RELAY bit set plus higher, unknown bits.
        obj.insert(
            "features".into(),
            serde_json::json!(features::RELAY | (1u64 << 40)),
        );
        obj.insert("a_future_field".into(), serde_json::json!({"nested": true}));
        obj.insert("another_future_list".into(), serde_json::json!([1, 2, 3]));

        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert!(cap.relay_capable);
        // We act on the common subset: we understand RELAY even though the peer
        // also advertises bits we don't.
        assert!(features::supports(cap.features, features::RELAY));
    }

    #[test]
    fn current_capability_round_trips() {
        let mut v = base_fields();
        let obj = v.as_object_mut().unwrap();
        obj.insert("relay_capable".into(), serde_json::json!(true));
        obj.insert(
            "protocol_version".into(),
            serde_json::json!(PROTOCOL_VERSION),
        );
        obj.insert("features".into(), serde_json::json!(features::ALL));
        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        let round: NodeCapability =
            serde_json::from_str(&serde_json::to_string(&cap).unwrap()).unwrap();
        assert_eq!(round.features, features::ALL);
        assert_eq!(round.protocol_version, PROTOCOL_VERSION);
        assert!(round.relay_capable);
    }
}

#[cfg(test)]
mod resident_ranges_tests {
    use super::version_compat_tests::base_fields;
    use super::*;

    /// A residency entry from a build before FUTURE_WORK #99 carries a count
    /// and no ranges, and must still parse — to NO ranges, which a reader
    /// takes as "priced by the count, as before". And one that carries ranges
    /// must survive the wire exactly, `releasable_layers` included, because a
    /// peer's plan is priced off it.
    #[test]
    fn a_residency_entry_with_or_without_ranges_reads_as_it_was_sent() {
        let mut v = base_fields();
        v.as_object_mut().unwrap().insert(
            "resident_layers".into(),
            serde_json::json!([{ "model_id": "glm", "layers": 12 }]),
        );
        let old: NodeCapability = serde_json::from_value(v).unwrap();
        assert_eq!(old.resident_layers.len(), 1);
        assert_eq!(old.resident_layers[0].layers, 12);
        assert!(
            old.resident_layers[0].ranges.is_empty(),
            "an older build's entry has no ranges, and must not be given any"
        );

        let entry = ResidentModelLayers {
            model_id: "glm".into(),
            layers: 14,
            ranges: vec![
                ResidentLayerRange {
                    start: 2,
                    end: 14,
                    releasable_layers: 0,
                },
                ResidentLayerRange {
                    start: 20,
                    end: 22,
                    releasable_layers: 2,
                },
            ],
        };
        let back: ResidentModelLayers =
            serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
        assert_eq!(back, entry);

        // A range whose sender left `releasable_layers` out releases nothing —
        // the side that cannot over-credit a peer.
        let partial: ResidentLayerRange =
            serde_json::from_value(serde_json::json!({ "start": 2, "end": 14 })).unwrap();
        assert_eq!(partial.releasable_layers, 0);
    }
}

#[cfg(test)]
mod serving_inference_tests {
    use super::version_compat_tests::base_fields;
    use super::*;

    /// **A node that says nothing about serving is willing to serve.**
    ///
    /// `can_serve_inference` was added after v0.3.189, so every node already in
    /// the swarm advertises a capability without it. If the missing field read
    /// as `false`, a coordinator on the new build would exclude every peer on
    /// an older one from inference — the whole swarm, on the day of release,
    /// and silently, because each such node is healthy and simply never chosen.
    ///
    /// This is `PeerResidency::WarmAmountUnknown`'s rule in a second place:
    /// silence is unknown, and unknown is never a refusal.
    #[test]
    fn a_capability_without_the_field_can_still_serve() {
        let v = base_fields();
        assert!(
            v.as_object().unwrap().get("can_serve_inference").is_none(),
            "this fixture must NOT carry the field — it stands in for a node \
             built before it existed"
        );
        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert!(
            cap.can_serve_inference,
            "a peer that has never heard of this field must remain a candidate"
        );
    }

    /// And an explicit refusal survives the wire, in both directions — the
    /// withdrawal is worthless if it does not reach the peers doing the routing.
    #[test]
    fn an_explicit_refusal_round_trips() {
        let mut v = base_fields();
        v.as_object_mut()
            .unwrap()
            .insert("can_serve_inference".into(), serde_json::json!(false));

        let cap: NodeCapability = serde_json::from_value(v).unwrap();
        assert!(!cap.can_serve_inference);

        let round: NodeCapability =
            serde_json::from_str(&serde_json::to_string(&cap).unwrap()).unwrap();
        assert!(
            !round.can_serve_inference,
            "the refusal must not be dropped when we re-serialise it — this \
             struct is re-broadcast, so a lost `false` reads as a recovery"
        );
    }
}
