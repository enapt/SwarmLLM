//! Wire-level P2P protocol messages: SwarmMessage, commands, rebalancing.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::credits::{CreditGossip, CreditTransaction};
use crate::identity::NicknameGossip;
use crate::ids::{ModelId, NodeId, ShardId};
use crate::inference::{
    InferenceError, InferenceRequest, LayerForward, LayerResult, PipelineAssignment,
    RemoteGenerateRequest, StreamingToken, TpAllReduceRequest, TpAllReduceResponse, TpRingChunk,
    VisionEncodeRequest, VisionEncodeResponse,
};
use crate::model::ModelManifest;
use crate::node::{NodeCapability, PeerExchangeResponse};
use crate::pool::PoolMessage;

/// A SwarmMessage wrapped with transport-authenticated sender identity.
/// The `sender` field is set by NetworkManager from the Noise-authenticated PeerId
/// (for request_response) or from the Ed25519-verified gossip signature.
/// This allows the dispatch handler to verify that message-internal sender claims
/// (e.g., ShardAnnounce.node_id) match the actual authenticated sender.
#[derive(Clone, Debug)]
pub struct AuthenticatedMessage {
    /// Transport-authenticated sender NodeId. None only for locally-generated messages.
    pub sender: Option<NodeId>,
    pub message: SwarmMessage,
}

/// Top-level enum for all protocol messages sent over libp2p.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SwarmMessage {
    // Discovery
    ShardAnnounce(ShardAnnounce),
    NodeCapabilityUpdate(NodeCapability),

    // Inference pipeline
    InferenceRequest(InferenceRequest),
    PipelineAssignment(PipelineAssignment),
    LayerForward(LayerForward),
    LayerResult(LayerResult),
    InferenceError(InferenceError),

    // Model manifest distribution
    ModelManifest(ModelManifest),

    // Credits
    CreditTransaction(CreditTransaction),

    // Health
    HealthPing {
        nonce: u64,
        timestamp: u64,
        /// Sender's node ID (for updating peer load in registry).
        #[serde(default)]
        node_id: Option<NodeId>,
        /// Number of active inference requests on the sender.
        #[serde(default)]
        active_request_count: u32,
    },
    HealthPong {
        nonce: u64,
        timestamp: u64,
        /// Sender's node ID (for updating peer load in registry).
        #[serde(default)]
        node_id: Option<NodeId>,
        /// Number of active inference requests on the sender.
        #[serde(default)]
        active_request_count: u32,
    },

    // Credits — gossip
    CreditGossip(CreditGossip),

    // Identity
    NicknameGossip(NicknameGossip),

    // Device Pools
    PoolMessage(PoolMessage),

    // Streaming — incremental token from the final pipeline node back to the originator
    StreamingToken(StreamingToken),

    // Shard download progress — broadcast so other nodes see downloads in progress
    ShardDownloadProgress(ShardDownloadProgress),

    // HuggingFace source gossip — tells peers where to download model shards from HF
    HfSourceGossip(HfSourceGossip),

    // Forward secrecy — ephemeral ECDH key exchange
    EphemeralKeyExchange(EphemeralKeyExchange),

    // Peer Exchange (PEX) — exchange known peer addresses on connection
    PeerExchangeRequest,
    PeerExchangeResponse(PeerExchangeResponse),

    // Vision — distributed mmproj encoding
    VisionEncodeRequest(VisionEncodeRequest),
    VisionEncodeResponse(VisionEncodeResponse),

    // Remote-generate fast path for single-segment distributed inference
    RemoteGenerateRequest(RemoteGenerateRequest),

    // Tensor Parallelism — AllReduce coordination
    TpAllReduceRequest(TpAllReduceRequest),
    TpAllReduceResponse(TpAllReduceResponse),
    /// Ring AllReduce: chunk sent between adjacent TP ranks.
    TpRingChunk(TpRingChunk),

    // Geo-aware regional gossip — Phase 18
    RegionShardSummary(RegionShardSummary),
    ModelDemandGossip(ModelDemandGossip),

    // Cross-node prefix-cache sharing (Item 8 Phase 1)
    PrefixCacheAnnounce(PrefixCacheAnnounce),

    // Cross-node inference cancellation. Sent by the request originator to
    // every peer involved in the pipeline when the caller flips
    // `InferenceRequest.cancel` or hangs up the SSE stream. Remote dispatch
    // resolves `request_id` against its `active_pipelines` map and signals
    // the worker subprocess to bail on the next decode iteration.
    CancelInference(CancelInference),

    // R130 — cross-pool / cross-node wishlist coordination. Opt-in publisher;
    // every node that opts in periodically gossips its top-K wishlist entries
    // on the existing regions topic. Receivers blend foreign interest into
    // their own wishlist score as a soft boost. Privacy-sensitive (publishes
    // what THIS node wants) so default-off.
    WishlistAnnouncement(WishlistAnnouncement),

    // R134 — inter-pool model availability. Pool owners opt in via
    // `pool.share_model_catalog` and announce which models their pool
    // serves. Receivers cache as a discovery signal — surfaces in the
    // admin UI as "Pool X also serves Y". Wire format ONLY; cross-pool
    // routing is a separate design decision still pending discussion.
    PoolModelAvailability(PoolModelAvailability),

    // NETWORKING_PLAN Phase 1 — application-level inference relay. When two
    // nodes cannot form a direct connection (both NAT'd, no hole-punch), an
    // inference message is routed through a mutually-reachable relay peer
    // (typically the anchor). The relay routes on the cleartext `relay_to`
    // header but never sees the plaintext inner message (end-to-end sealed
    // for `relay_to`). Additive: an older node that can't deserialize this
    // variant simply never receives one — the sender gates on the peer's
    // advertised capability before wrapping.
    RelayedEnvelope(RelayedEnvelope),

    // A coordinator asking the node serving a remote-generate reply to send
    // a range of its content tokens again (gotcha #438). Additive: gated on
    // `features::RESEND_TOKENS`, so an older node is never sent one.
    ResendTokens(ResendTokens),
}

/// NETWORKING_PLAN Phase 1 — application-level inference relay envelope.
///
/// Routed through a mutually-reachable relay peer (usually the anchor) when a
/// direct connection between two NAT'd nodes cannot be formed. The relay is a
/// DUMB PIPE: it forwards based on the cleartext `relay_to` header, but the
/// `sealed` inner `SwarmMessage` is ephemeral-sealed end-to-end for
/// `relay_to`'s X25519 key (derived from its NodeId), so a middle relay never
/// sees the prompt or the streamed tokens — preserving the Layer-1 encryption
/// invariant that only the request's true endpoints read cleartext.
///
/// The seal AAD binds `origin || relay_to || request_id`, so a relay cannot
/// re-address, replay, or cross-wire an envelope without failing Poly1305.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelayedEnvelope {
    /// Final recipient. The relay forwards to this node; only this node holds
    /// the X25519 secret that opens `sealed`.
    pub relay_to: NodeId,
    /// Original author of the inner message. The recipient derives the
    /// decryption key from it and learns a reverse route ("reach `origin` via
    /// the peer this arrived from"). A relay MUST refuse to forward unless the
    /// transport-authenticated sender equals `origin` — this bounds relaying
    /// to a single hop and blocks loops / traffic amplification.
    pub origin: NodeId,
    /// Correlation id (the inner message's request_id where one exists), bound
    /// into the seal AAD. Lets the relay rate-limit and endpoints correlate.
    pub request_id: uuid::Uuid,
    /// Ephemeral X25519 public key for the per-message ECDH (`ephemeral_seal`).
    pub ephemeral_pub: [u8; 32],
    /// ChaCha20-Poly1305-sealed inner `SwarmMessage` (serde_json bytes).
    pub sealed: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CancelInference {
    pub request_id: uuid::Uuid,
}

/// Ask the node serving a remote-generate reply to send some of its content
/// tokens again. Sent by the coordinator when the reply's token sequence has a
/// hole — tokens numbered past `from_token_id` have arrived and the ones from
/// `from_token_id` up to (not including) `to_token_id` have not.
///
/// Gated on `features::RESEND_TOKENS`: a serving node that advertises it keeps
/// the reply's tokens for a while after sending them (see the daemon's retained
/// replies), so filling a hole costs one round trip instead of the reply's
/// tail. The serving node re-sends the terminal token too when the reply has
/// finished, since that is the frame the requester is waiting on.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResendTokens {
    pub request_id: uuid::Uuid,
    /// First content token wanted (inclusive), as numbered by the sender.
    pub from_token_id: u32,
    /// One past the last content token wanted. Bounded by the serving node.
    pub to_token_id: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardAnnounce {
    pub node_id: NodeId,
    pub shards: Vec<ShardId>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Models for which `shards` is the sender's COMPLETE set.
    ///
    /// A receiver drops any holder record it has for this node under these
    /// models that is absent from `shards`. Models not listed here are only
    /// added to, never retracted.
    ///
    /// This field exists because the same message is sent with three different
    /// scopes — the whole local shard set (periodic re-announce), one model's
    /// remaining shards (shard deletion), and a single newly-acquired shard —
    /// and a receiver cannot tell them apart. Without it the handler had to
    /// assume the additive case, so nothing could ever retract: a peer that
    /// deleted shards kept its claim on them, and `delete_model`'s
    /// `shards: vec![]` "we no longer host anything" broadcast decoded to a
    /// loop over zero elements and did nothing at all.
    ///
    /// `#[serde(default)]` keeps this compatible with older nodes: their
    /// announcements arrive with an empty list and stay purely additive, which
    /// is exactly the pre-existing behaviour.
    #[serde(default)]
    pub complete_for_models: Vec<ModelId>,
    /// Build tag for each entry of `shards`, positionally parallel to it.
    ///
    /// A model id is derived from a display name (`slugify_model_name`), so
    /// every independent GGUF build of one model collapses into one identity —
    /// three Q4_K_M builds of Qwen2.5-Coder-7B were live on the swarm at once,
    /// within 800 bytes of each other, sharing not one shard hash. Holder
    /// records keyed on `ShardId` alone therefore pool nodes holding DIFFERENT
    /// FILES, and the scheduler routes to either: correctness is safe (every
    /// shard is hash-verified before load) but the fetch is a guaranteed
    /// wasted transfer.
    ///
    /// This is the sender's own content hash for that shard, truncated by
    /// `build_tag_from_hash` — see there for why a *per-shard* hash and not
    /// the manifest hash.
    ///
    /// `#[serde(default)]` keeps this compatible with older nodes: their
    /// announcements arrive with an empty vector, which reads as
    /// `BUILD_TAG_UNKNOWN` for every shard and is exactly the pre-existing
    /// behaviour. A length that does not match `shards` is treated the same
    /// way, wholesale — a positional field that has desynced says nothing
    /// trustworthy about any position, so guessing per-index is worse than
    /// admitting ignorance.
    #[serde(default)]
    pub shard_builds: Vec<u64>,
}

/// "I could not say which build this is." Never compare equal to anything,
/// including itself — an unknown tag must never *exclude* a holder, the same
/// contract `max_hostable_layers` keeps for unknown capacity.
pub const BUILD_TAG_UNKNOWN: u64 = 0;

/// The build identity of one shard, as carried on the wire.
///
/// **Per-shard content hash, deliberately not the manifest hash.** A manifest
/// hash is recomputed locally whenever `merge_known_shard_hashes` recovers a
/// hash the sender did not have, so two nodes holding the SAME build can carry
/// different manifest hashes depending on what each has learned — which would
/// make it a source of false mismatches. A shard's content hash is a property
/// of the bytes and nothing else, so two holders of one build always agree.
///
/// The first 8 bytes of BLAKE3 are ample: this separates concurrent builds of
/// one model, it is not a security boundary (the full hash is still verified
/// on load), and a collision costs one wasted transfer — exactly what happens
/// today for every such holder.
///
/// An all-zero hash means "not known" in a manifest (`build_shard_infos_from_
/// layouts` writes zeros for shards its author does not hold), and maps to
/// `BUILD_TAG_UNKNOWN` here rather than to a real tag of zero.
pub fn build_tag_from_hash(hash: &[u8; 32]) -> u64 {
    if hash.iter().all(|b| *b == 0) {
        return BUILD_TAG_UNKNOWN;
    }
    let tag = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ]);
    // Vanishingly unlikely, but a real hash must never render as "unknown".
    if tag == BUILD_TAG_UNKNOWN {
        1
    } else {
        tag
    }
}

/// Do a known-expected and a claimed build tag positively DISAGREE?
///
/// Unknown on either side is not a disagreement. This is the single place that
/// decides, so a caller cannot accidentally read "unknown" as "wrong".
pub fn build_tags_conflict(expected: u64, claimed: u64) -> bool {
    expected != BUILD_TAG_UNKNOWN && claimed != BUILD_TAG_UNKNOWN && expected != claimed
}

/// State of a shard download.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DownloadState {
    Queued,
    Downloading,
    Verifying,
    Complete,
    Failed,
}

impl fmt::Display for DownloadState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Queued => write!(f, "queued"),
            Self::Downloading => write!(f, "downloading"),
            Self::Verifying => write!(f, "verifying"),
            Self::Complete => write!(f, "complete"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

/// Shard download progress broadcast — lets other nodes see downloads in real time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardDownloadProgress {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    /// 0-100 percentage
    pub progress_pct: u32,
    pub state: DownloadState,
}

/// HuggingFace source gossip — tells peers where to download a model's shards from HF CDN.
/// Without this, only the node that originally downloaded from HF knows the repo_id/filename.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfSourceGossip {
    pub model_id: ModelId,
    pub repo_id: String,
    pub filename: String,
    pub publisher: NodeId,
    /// Filename of the mmproj GGUF on HuggingFace (for VLM models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_filename: Option<String>,
}

/// Regional shard summary — compact per-region model availability gossip.
/// Published periodically by each node for its region on `swarm/regions`.
/// O(regions * models) entries (~1KB each), enabling geo-aware auto-manage
/// decisions without iterating all holders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegionShardSummary {
    /// ISO 3166-1 alpha-2 region code (e.g. "US", "DE").
    pub region: String,
    pub model_id: ModelId,
    /// Per-shard holder count in this region: (shard_index, holder_count).
    pub shard_counts: Vec<(u32, u32)>,
    /// Total nodes in this region (as seen by the publisher).
    pub region_node_count: u32,
    pub publisher: NodeId,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// R130 — wishlist announcement. Publisher's top-K wishlist entries with a
/// coarse score signal. Receivers aggregate per-model interest across
/// publishers and apply a small boost in their own wishlist score. The
/// model_id field is the canonical identifier shared across the swarm; we
/// do NOT leak hostnames, pool composition, or per-shard interest — only
/// "we'd like this model" at the model granularity.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WishlistAnnouncement {
    pub publisher: NodeId,
    /// Highest-scoring entries from the publisher's local wishlist
    /// (bounded; see `MAX_WISHLIST_ANNOUNCE_ENTRIES` on the publishing
    /// side for the cap). Smaller cap on the wire than the local
    /// wishlist itself — gossip carries the headline, not the full list.
    pub entries: Vec<WishlistAnnouncementEntry>,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WishlistAnnouncementEntry {
    pub model_id: ModelId,
    /// Publisher's local wishlist score (0..100). Receivers treat this
    /// purely as a ranking signal; absolute values aren't comparable
    /// across nodes (different swarm context, different VRAM, etc.).
    pub score: u32,
}

/// R134: inter-pool model availability announcement. When a pool opts
/// in via `pool.share_model_catalog`, the owner periodically broadcasts
/// the model_ids the pool can serve. Outsiders cache this as a
/// "discovery" signal — which pools claim to serve which models — but
/// nothing about pool composition, member count, or per-shard interest
/// is exposed. A k-anonymity floor on the sender side blocks pools
/// smaller than `MIN_K_ANONYMITY_MEMBERS` from announcing, so the
/// channel cannot be used to enumerate small private pools.
///
/// Routing across pool boundaries based on this signal is explicitly
/// out of scope for the wire format — the private-mode contract that
/// "your inference stays in your pool" is preserved. Cross-pool
/// routing is a separate design decision that needs its own opt-in,
/// trust model, and UI surface.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolModelAvailability {
    /// Pool ID = pool owner's NodeId. Receivers verify the
    /// `owner_signature` against this key.
    pub pool_id: NodeId,
    /// Model IDs the pool can serve. Capped at
    /// `MAX_POOL_MODEL_ANNOUNCE_ENTRIES` on the wire to keep gossip
    /// bounded; senders rank by recent local-host activity.
    pub model_ids: Vec<ModelId>,
    /// Milliseconds since Unix epoch. Receivers apply the standard
    /// one-sided staleness window to defeat replay.
    pub timestamp_ms: u64,
    /// Ed25519 signature over the gossip payload (see
    /// `pool::crypto::pool_model_availability_payload`).
    pub owner_signature: Vec<u8>,
}

/// Model demand gossip — EMA-smoothed request rate per model per region.
/// Published alongside RegionShardSummary to inform replication targets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelDemandGossip {
    pub model_id: ModelId,
    /// ISO 3166-1 alpha-2 region code.
    pub region: String,
    /// Exponentially decayed request rate (requests per 10-minute window).
    pub decayed_rate: f64,
    /// Raw request count in the latest window (before decay).
    pub window_requests: u64,
    pub publisher: NodeId,
    /// Milliseconds since Unix epoch.
    pub timestamp_ms: u64,
}

/// Ephemeral ECDH key exchange message for forward secrecy.
///
/// When a node wants to establish a forward-secret session, it sends this message
/// containing an ephemeral X25519 public key. The recipient generates its own
/// ephemeral key, derives the session key from the DH, and replies with its
/// ephemeral public key.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EphemeralKeyExchange {
    /// Unique session identifier.
    pub session_id: uuid::Uuid,
    /// The sender's node identity.
    pub node_id: NodeId,
    /// Ephemeral X25519 public key (generated fresh for this session).
    pub ephemeral_pubkey: [u8; 32],
    /// Whether this is an initiation (true) or a response (false).
    pub is_initiator: bool,
}

/// Commands sent from daemon tasks to the NetworkManager.
///
/// `Broadcast` wraps a `SwarmMessage` for GossipSub. `SendTensor` and
/// `SendTensorResult` route tensor data through the unified request_response
/// codec with binary type-tag framing (WIRE_TAG_TENSOR = 0x01).
#[derive(Clone, Debug)]
pub enum NetworkCommand {
    /// Broadcast a message via GossipSub to all subscribers.
    Broadcast(SwarmMessage),
    /// Send a tensor forward pass to a specific peer via binary type-tag codec.
    SendTensor {
        target_peer_bytes: Vec<u8>,
        forward: LayerForward,
    },
    /// Internal: post-encrypt `SendTensor` continuation. The encrypt+encode
    /// step in `handle_send_tensor` is offloaded to a `tokio::spawn` task so
    /// the NetworkManager's event loop is not blocked for ~50–200µs/forward
    /// on ChaCha20 sealing of large activations. The spawned task posts
    /// this variant back through `internal_cmd_tx` once the wire-ready
    /// payload is in hand; the handler then performs only the synchronous
    /// `send_request` + `pending_tensor_outbound` bookkeeping on the
    /// critical task. Not constructed by daemon-side callers — they keep
    /// sending the plain `SendTensor` variant.
    SendEncodedTensor {
        target_peer_bytes: Vec<u8>,
        payload: Vec<u8>,
        request_id: uuid::Uuid,
        num_layers: u32,
        activation_bytes: usize,
    },
    /// Send a tensor result back to a specific peer as a new request.
    SendTensorResult {
        target_peer_bytes: Vec<u8>,
        result: LayerResult,
    },
    /// Send a streaming token to a specific peer (originator of the request).
    SendStreamingToken {
        target_peer_bytes: Vec<u8>,
        token: StreamingToken,
    },
    /// Send a shard transfer request to a specific peer.
    SendShardRequest {
        target_peer_bytes: Vec<u8>,
        request: ShardRequest,
    },
    /// Send an AllReduce partial to the TP coordinator.
    SendAllReduceRequest {
        target_peer_bytes: Vec<u8>,
        request: TpAllReduceRequest,
    },
    /// Send an AllReduce response to a specific TP rank.
    SendAllReduceResponse {
        target_peer_bytes: Vec<u8>,
        response: TpAllReduceResponse,
    },
    /// Send a ring AllReduce chunk to the right neighbor in the ring.
    SendRingChunk {
        target_peer_bytes: Vec<u8>,
        chunk: TpRingChunk,
    },
    /// Send a SwarmMessage directly to a specific peer via request_response.
    ///
    /// `delivery_request_id`, when set, opts into ACK-timeout tracking: if
    /// the receiver doesn't ACK within a few seconds, the daemon closes
    /// `streaming_token_txs[delivery_request_id]` so the caller fails fast
    /// instead of waiting the full FIRST_TOKEN_TIMEOUT (120s). Used by the
    /// remote-generate fast path to detect rare libp2p rr silent-drops.
    SendDirectMessage {
        target_peer_bytes: Vec<u8>,
        message: SwarmMessage,
        delivery_request_id: Option<uuid::Uuid>,
    },
    /// Dial a multiaddr to connect to a new peer.
    DialAddress(String),
    /// S5: Register as a Kademlia provider for the given shards.
    /// Called on shard acquisition (download complete, startup scan).
    StartProviding(Vec<ShardId>),
    /// S5: Stop providing the given shards via Kademlia.
    /// Called on shard deletion.
    StopProviding(Vec<ShardId>),
    /// Item 8 Phase 2: send a cross-node prefix KV fetch to `target_peer`.
    /// The daemon caller installs a `tokio::sync::oneshot::Sender<Option<Vec<u8>>>`
    /// in `SharedState` keyed by `request_id` BEFORE sending this command;
    /// NetworkManager resolves it with the payload bytes on
    /// `SwarmResponse::PrefixKvData` arrival (or `None` on failure/timeout).
    SendPrefixKvFetch {
        target_peer_bytes: Vec<u8>,
        request_id: uuid::Uuid,
        model_id: ModelId,
        block_hash: [u8; 32],
    },
    /// Item 8 Phase 2b: deliver an inbound-served PrefixKv response. The
    /// serving task extracts a snapshot from the local worker and posts
    /// this command; NetworkManager looks up the stored ResponseChannel
    /// by `ticket` and emits the reply. `payload=None` produces a miss
    /// reply; `Some(bytes)` produces a hit reply.
    DeliverPrefixKvResponse {
        ticket: uuid::Uuid,
        request_id: uuid::Uuid,
        payload: Option<Vec<u8>>,
    },
    /// Mirror of `DeliverPrefixKvResponse` for shard-transfer serving. The
    /// serving task does the disk read + bandwidth-throttle sleep off the
    /// swarm event loop and posts this command with the produced bytes.
    /// NetworkManager looks up the stored ResponseChannel by `ticket`,
    /// constructs `SwarmResponse::ShardData { data, total_size }`, emits
    /// the reply, and bumps the `shard_bytes_served` atomic for credit
    /// seeding.
    DeliverShardResponse {
        ticket: uuid::Uuid,
        data: Vec<u8>,
        total_size: u64,
    },
}

/// Events that trigger shard rebalancing.
#[derive(Clone, Debug)]
pub enum RebalanceEvent {
    PeerLeft(NodeId),
}

// ---- Shard transfer protocol ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardRequest {
    pub shard_id: ShardId,
    pub chunk_offset: u64,
    pub chunk_size: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardResponse {
    pub data: Vec<u8>,
    pub total_size: u64,
}

impl ShardResponse {
    /// Build the canonical empty/error response — used for refused
    /// requests, queue-full rejections, and disk read/seek/open failures
    /// (R97). Centralised so adding an error field can't drift across
    /// the 8+ rejection sites in `network::manager::{requests,shard_transfer}`.
    pub fn empty() -> Self {
        Self {
            data: Vec::new(),
            total_size: 0,
        }
    }
}

/// One block in a `PrefixCacheAnnounce`. The hash is a chained BLAKE3
/// rollup over the token IDs of the prompt prefix that produced this KV
/// snapshot, with `token_count` total tokens covered.
///
/// Hash chain: `block_hash[0] = blake3(u32_le(tokens[0..B]))`,
/// `block_hash[i] = blake3(block_hash[i-1] || u32_le(tokens[i*B..(i+1)*B]))`.
/// Two prompts that share the first `i` blocks (under the same block size
/// `B`) have identical `block_hash[0..i]`, so longest-prefix match by hash
/// works without ever transmitting tokens.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefixBlockEntry {
    /// Chained BLAKE3 hash uniquely identifying this token-prefix block.
    pub block_hash: [u8; 32],
    /// Number of prompt tokens covered by the prefix this block ends at
    /// (i.e. the strict prefix length the snapshot represents).
    pub token_count: u32,
}

/// Cross-node prefix cache announcement (Item 8 Phase 1).
///
/// Each worker broadcasts the BLAKE3 hashes of cached prompt-prefix blocks
/// it currently holds for a given model. Receivers build a per-model
/// `block_hash → {peer_id}` index that lets them, in Phase 2+, fetch a
/// pre-computed KV snapshot from the announcing peer instead of re-running
/// prefill.
///
/// Announcements supersede prior announcements from the same `(node_id,
/// model_id)` pair — the receiver replaces all prior entries for that pair
/// with the new `blocks` set, so the index reflects the announcer's
/// current cache state (including evictions).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixCacheAnnounce {
    pub node_id: NodeId,
    pub model_id: ModelId,
    pub blocks: Vec<PrefixBlockEntry>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Event emitted when auto-manage prunes (deletes) an over-replicated shard.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PruneEvent {
    pub model_id: ModelId,
    pub model_name: String,
    pub shard_index: u32,
    pub reason: String,
    pub freed_bytes: u64,
    pub remaining_local_shards: u32,
    pub holder_count_before: usize,
    pub holder_count_after: usize,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod build_tag_tests {
    use super::*;
    use crate::ids::{ModelId, NodeId, ShardId};

    /// The shape an older node sends: no `shard_builds` key at all.
    ///
    /// The protocol rule is that a new field is not done until the pair has
    /// been run in BOTH directions — the swarm is always mixed-version during
    /// a rollout, and there is no second chance once an announcement fails to
    /// decode.
    #[test]
    fn an_older_nodes_announcement_still_decodes() {
        let older = serde_json::json!({
            "node_id": NodeId([1u8; 32]),
            "shards": [ShardId { model_id: ModelId("m".into()), index: 0 }],
            "timestamp": chrono::Utc::now(),
            "complete_for_models": [],
        });
        let decoded: ShardAnnounce =
            serde_json::from_value(older).expect("an older announcement must still decode");
        assert!(
            decoded.shard_builds.is_empty(),
            "and must read as 'no build stated', which the receiver treats as unknown"
        );
    }

    /// The other direction: a NEWER node's announcement reaching an older one.
    /// `ShardAnnounce` carries no `deny_unknown_fields`, so the extra key is
    /// ignored — reproduced here against a struct with the pre-change shape,
    /// because "serde ignores unknown fields" is the kind of assumption that
    /// is cheap to check and expensive to be wrong about.
    #[test]
    fn a_newer_announcement_decodes_on_a_node_that_predates_the_field() {
        #[derive(serde::Deserialize)]
        #[allow(dead_code)]
        struct OldShardAnnounce {
            node_id: NodeId,
            shards: Vec<ShardId>,
            timestamp: chrono::DateTime<chrono::Utc>,
            #[serde(default)]
            complete_for_models: Vec<ModelId>,
        }

        let newer = ShardAnnounce {
            node_id: NodeId([1u8; 32]),
            shards: vec![ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            }],
            timestamp: chrono::Utc::now(),
            complete_for_models: vec![],
            shard_builds: vec![build_tag_from_hash(&[7u8; 32])],
        };
        let wire = serde_json::to_string(&newer).expect("serialize");
        let old: OldShardAnnounce =
            serde_json::from_str(&wire).expect("an older node must not choke on the new field");
        assert_eq!(old.shards.len(), 1);
    }

    /// Two builds of one model must be distinguishable, and one build must
    /// look the same from every holder of it.
    #[test]
    fn a_tag_identifies_the_bytes_and_nothing_else() {
        let a = build_tag_from_hash(&[7u8; 32]);
        let b = build_tag_from_hash(&[8u8; 32]);
        assert_ne!(a, b);
        assert_eq!(a, build_tag_from_hash(&[7u8; 32]));
        assert_ne!(a, BUILD_TAG_UNKNOWN);
        assert_eq!(build_tag_from_hash(&[0u8; 32]), BUILD_TAG_UNKNOWN);
    }
}
