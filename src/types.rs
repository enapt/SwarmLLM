use serde::{Deserialize, Serialize};
use std::fmt;

// ---- Identity ----
/// Wrapper around Ed25519 public key. This IS the node's identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub [u8; 32]);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", hex::encode(&self.0[..8]))
    }
}

// ---- Models ----
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: ModelId,
    pub name: String,
    pub architecture: ModelArchitecture,
    pub num_layers: u32,
    pub num_params_billions: f32,
    pub quantization: Quantization,
    pub total_size_bytes: u64,
    pub shard_count: u32,
    pub shards: Vec<ShardInfo>,
    pub tokenizer_hash: Blake3Hash,
    pub manifest_hash: Blake3Hash,
    pub publisher: NodeId,
    pub publish_date: chrono::DateTime<chrono::Utc>,
    pub license: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Mixtral {
        num_experts: u32,
        experts_per_token: u32,
    },
    Qwen2,
    DeepSeek {
        num_experts: u32,
        experts_per_token: u32,
    },
    Phi,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Quantization {
    Q4KM,
    Q5KM,
    Q6K,
    Q8_0,
    FP16,
}

// ---- Shards ----
pub type Blake3Hash = [u8; 32];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardInfo {
    pub index: u32,
    pub layer_range: (u32, u32),
    pub size_bytes: u64,
    pub hash: Blake3Hash,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ShardId {
    pub model_id: ModelId,
    pub index: u32,
}

// ---- Node Capabilities ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapability {
    pub node_id: NodeId,
    pub gpu: Option<GpuInfo>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub disk_available_mb: u64,
    pub bandwidth_mbps: f32,
    pub hosted_shards: Vec<ShardId>,
    pub max_contribution: ContributionLevel,
    pub uptime_seconds: u64,
    pub version: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_mb: u64,
    pub vram_available_mb: u64,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ContributionLevel {
    Minimal,
    Moderate,
    Maximum,
}

// ---- Inference ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub id: uuid::Uuid,
    pub model_id: ModelId,
    pub messages: Vec<ChatMessage>,
    pub sampling_params: SamplingParams,
    pub stream: bool,
    pub requester: NodeId,
    pub priority: PriorityTier,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub max_tokens: u32,
    #[serde(default)]
    pub stop: Vec<String>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            max_tokens: 2048,
            stop: vec![],
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

// ---- Credits ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditBalance {
    pub node_id: NodeId,
    pub balance: i64,
    pub lifetime_earned: u64,
    pub lifetime_spent: u64,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PriorityTier {
    Bronze = 0,
    Silver = 1,
    Gold = 2,
    Platinum = 3,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditTransaction {
    pub id: uuid::Uuid,
    pub from: NodeId,
    pub to: NodeId,
    pub amount: i64,
    pub reason: TransactionReason,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature_from: Vec<u8>,
    pub signature_to: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TransactionReason {
    InferenceServed { request_id: uuid::Uuid, tokens: u32 },
    ShardHosting { shard_id: ShardId, hours: f32 },
    ShardSeeding { shard_id: ShardId, bytes: u64 },
    RelayService { duration_seconds: u64 },
    InferenceConsumed { request_id: uuid::Uuid, tokens: u32 },
    Penalty { reason: String },
}

// ---- Pipeline ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineAssignment {
    pub request_id: uuid::Uuid,
    pub segments: Vec<PipelineSegment>,
    pub standbys: Vec<PipelineSegment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PipelineSegment {
    pub node_id: NodeId,
    pub shard_id: ShardId,
    pub layer_range: (u32, u32),
}

// ---- Network Messages ----
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
    HealthPing { nonce: u64, timestamp: u64 },
    HealthPong { nonce: u64, timestamp: u64 },

    // Credits — gossip
    CreditGossip(CreditGossip),

    // Governance
    ModelVote(ModelVote),

    // Self-governance (Phase 7)
    Proposal(Proposal),
    ProposalAmendment(ProposalAmendment),
    ProposalStatusChange(ProposalStatusChange),
    ProposalVote(ProposalVote),
    Issue(Issue),
    IssueComment(IssueComment),
    IssueStatusChange(IssueStatusChange),
    IssueUpvote(IssueUpvote),
    ReleaseCandidate(ReleaseCandidate),
    TestReport(TestReport),
    ReleaseApproval(ReleaseApproval),
    ChangelogEntry(ChangelogEntry),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardAnnounce {
    pub node_id: NodeId,
    pub shards: Vec<ShardId>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerForward {
    pub request_id: uuid::Uuid,
    pub sequence_num: u32,
    pub activations: Vec<u8>,
    pub format: TensorFormat,
    /// Populated locally after receiving from the network — not serialized over the wire.
    /// Contains the libp2p PeerId bytes of the sender so we can route the result back.
    #[serde(skip)]
    pub sender_peer_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TensorFormat {
    FP16,
    FP32,
    INT8,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerResult {
    pub request_id: uuid::Uuid,
    pub token_ids: Vec<u32>,
    pub finish_reason: Option<NetworkFinishReason>,
    /// Intermediate hidden-state activations (for non-final pipeline segments).
    /// Empty for the final segment (which returns token_ids instead).
    #[serde(default)]
    pub activations: Vec<u8>,
}

/// Finish reason for network protocol messages (distinct from inference::executor::FinishReason).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NetworkFinishReason {
    Stop,
    MaxTokens,
    Error(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceError {
    pub request_id: uuid::Uuid,
    pub error: String,
    pub recoverable: bool,
}

/// Bucketed credit balance gossip for network-wide percentile estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditGossip {
    pub node_id: NodeId,
    pub balance_bucket: i64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelVote {
    pub voter: NodeId,
    pub model_manifest_hash: Blake3Hash,
    pub vote: bool,
    pub weight: u64,
    pub signature: Vec<u8>,
}

// ---- Governance ----

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum GovernanceRole {
    Member,
    Contributor,
    Maintainer,
    Council,
}

impl GovernanceRole {
    pub fn can_create_proposals(&self) -> bool {
        matches!(
            self,
            GovernanceRole::Contributor | GovernanceRole::Maintainer | GovernanceRole::Council
        )
    }

    pub fn can_approve_releases(&self) -> bool {
        matches!(self, GovernanceRole::Maintainer | GovernanceRole::Council)
    }

    pub fn can_emergency_veto(&self) -> bool {
        matches!(self, GovernanceRole::Council)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GovernanceParams {
    // Voting
    pub code_change_voting_days: u32,
    pub protocol_change_voting_days: u32,
    pub governance_change_voting_days: u32,
    pub code_change_quorum_pct: f32,
    pub protocol_change_quorum_pct: f32,
    pub governance_change_quorum_pct: f32,
    pub code_change_approval_pct: f32,
    pub protocol_change_approval_pct: f32,
    pub governance_change_approval_pct: f32,
    // Roles
    pub contributor_percentile: f32,
    pub contributor_min_uptime_days: u32,
    pub maintainer_percentile: f32,
    pub maintainer_min_uptime_days: u32,
    pub maintainer_min_accepted_proposals: u32,
    pub council_seats: u32,
    pub council_term_days: u32,
    // Releases
    pub release_approval_threshold: u32,
    pub release_verification_days_stable: u32,
    pub release_verification_days_patch: u32,
    pub canary_phase1_days: u32,
    pub canary_phase2_pct: f32,
    pub canary_phase3_pct: f32,
    pub max_proposal_amendments: u32,
    // Issues
    pub min_credit_balance_for_issues: i64,
    pub issue_auto_close_days: u32,
}

impl Default for GovernanceParams {
    fn default() -> Self {
        Self {
            code_change_voting_days: 7,
            protocol_change_voting_days: 14,
            governance_change_voting_days: 21,
            code_change_quorum_pct: 0.10,
            protocol_change_quorum_pct: 0.20,
            governance_change_quorum_pct: 0.25,
            code_change_approval_pct: 0.50,
            protocol_change_approval_pct: 0.66,
            governance_change_approval_pct: 0.75,
            contributor_percentile: 0.80,
            contributor_min_uptime_days: 30,
            maintainer_percentile: 0.95,
            maintainer_min_uptime_days: 90,
            maintainer_min_accepted_proposals: 3,
            council_seats: 7,
            council_term_days: 365,
            release_approval_threshold: 3,
            release_verification_days_stable: 7,
            release_verification_days_patch: 3,
            canary_phase1_days: 3,
            canary_phase2_pct: 0.05,
            canary_phase3_pct: 0.25,
            max_proposal_amendments: 3,
            min_credit_balance_for_issues: 0,
            issue_auto_close_days: 90,
        }
    }
}

/// Network-wide statistics used for role calculation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NetworkStats {
    pub total_active_vote_weight: u64,
    pub nodes_with_30d_uptime: u32,
    pub total_active_nodes: u32,
}

/// Extended per-node stats used for governance role calculation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GovernanceNodeStats {
    pub lifetime_earned_percentile: f32,
    pub uptime_days: u32,
    pub accepted_proposals: u32,
    pub is_council_member: bool,
}

impl GovernanceRole {
    pub fn from_node_governance_stats(
        stats: &GovernanceNodeStats,
        params: &GovernanceParams,
    ) -> Self {
        if stats.is_council_member {
            GovernanceRole::Council
        } else if stats.lifetime_earned_percentile >= params.maintainer_percentile
            && stats.uptime_days >= params.maintainer_min_uptime_days
            && stats.accepted_proposals >= params.maintainer_min_accepted_proposals
        {
            GovernanceRole::Maintainer
        } else if stats.lifetime_earned_percentile >= params.contributor_percentile
            && stats.uptime_days >= params.contributor_min_uptime_days
        {
            GovernanceRole::Contributor
        } else {
            GovernanceRole::Member
        }
    }
}

// ---- Issues ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Issue {
    pub hash: Blake3Hash,
    pub author: NodeId,
    pub title: String,
    pub body: String,
    pub category: IssueCategory,
    pub severity: Option<IssueSeverity>,
    pub status: IssueStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub upvotes: u32,
    pub tags: Vec<String>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueCategory {
    Bug,
    FeatureRequest,
    Performance,
    Security,
    Documentation,
    ModelRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum IssueStatus {
    Open,
    Acknowledged,
    InProgress,
    Resolved,
    Closed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueComment {
    pub issue_hash: Blake3Hash,
    pub author: NodeId,
    pub body: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueUpvote {
    pub issue_hash: Blake3Hash,
    pub voter: NodeId,
    pub weight: u64,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IssueStatusChange {
    pub issue_hash: Blake3Hash,
    pub new_status: IssueStatus,
    pub changed_by: NodeId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

// ---- Proposals ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposal {
    pub hash: Blake3Hash,
    pub author: NodeId,
    pub title: String,
    pub body: String,
    pub category: ProposalCategory,
    pub status: ProposalStatus,
    pub linked_issues: Vec<Blake3Hash>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub voting_deadline: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
    pub patch: Option<ProposalPatch>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalCategory {
    CodeChange,
    ProtocolChange,
    GovernanceChange,
    ModelAddition,
    ModelDeprecation,
    ParameterTuning,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalPatch {
    pub diff_hash: Blake3Hash,
    pub diff_size_bytes: u64,
    pub files_changed: Vec<String>,
    pub insertions: u32,
    pub deletions: u32,
    pub inline_diff: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProposalStatus {
    Draft,
    Open,
    Amended,
    Accepted,
    Rejected,
    Implemented,
    Released,
    Withdrawn,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalVote {
    pub proposal_hash: Blake3Hash,
    pub voter: NodeId,
    pub vote: VoteChoice,
    pub weight: u64,
    pub role: GovernanceRole,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum VoteChoice {
    Approve,
    Reject,
    Abstain,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalAmendment {
    pub proposal_hash: Blake3Hash,
    pub author: NodeId,
    pub body: String,
    pub new_patch: Option<ProposalPatch>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalStatusChange {
    pub proposal_hash: Blake3Hash,
    pub new_status: ProposalStatus,
    pub changed_by: NodeId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

// ---- Releases ----

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemVer {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre: Option<String>,
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre {
            write!(f, "-{pre}")?;
        }
        Ok(())
    }
}

impl SemVer {
    pub fn is_prerelease(&self) -> bool {
        self.pre.is_some()
    }

    pub fn to_key(&self) -> String {
        format!("{self}")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Architecture {
    X86_64,
    Aarch64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseBinary {
    pub platform: Platform,
    pub arch: Architecture,
    pub hash: Blake3Hash,
    pub size_bytes: u64,
    pub shard_manifest: Blake3Hash,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseCandidate {
    pub version: SemVer,
    pub builder: NodeId,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub changelog: String,
    pub included_proposals: Vec<Blake3Hash>,
    pub binaries: Vec<ReleaseBinary>,
    pub source_hash: Blake3Hash,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReleaseApproval {
    pub release_version: SemVer,
    pub approver: NodeId,
    pub role: GovernanceRole,
    pub binary_hashes_verified: bool,
    pub test_suite_passed: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

// ---- Testing ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestReport {
    pub release_version: SemVer,
    pub tester: NodeId,
    pub platform: Platform,
    pub architecture: Architecture,
    pub gpu: Option<String>,
    pub status: TestStatus,
    pub results: TestResults,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TestStatus {
    Passed,
    Failed { failures: Vec<String> },
    HashMismatch { expected: String, actual: String },
    BuildFailed { error: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TestResults {
    pub tests_run: u32,
    pub tests_passed: u32,
    pub tests_failed: u32,
    pub tests_skipped: u32,
    pub build_time_seconds: u64,
    pub binary_hash: Blake3Hash,
    pub binary_size_bytes: u64,
}

// ---- Changelog ----

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub version: SemVer,
    pub date: chrono::DateTime<chrono::Utc>,
    pub entries: Vec<ChangelogItem>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChangelogItem {
    pub category: ProposalCategory,
    pub title: String,
    pub proposal_hash: Blake3Hash,
    pub author: NodeId,
}

/// Canary rollout phase.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CanaryPhase {
    /// Day 0-3: only auto_update="all" nodes
    Phase1,
    /// Day 3-7: 5% of auto_update="stable" nodes
    Phase2,
    /// Day 7-10: 25% of auto_update="stable" nodes
    Phase3,
    /// Day 10+: 100% of auto_update="stable" nodes
    Complete,
    /// Rollout halted due to issues
    Halted { reason: String },
}

// ---- Network Commands ----
/// Commands sent from daemon tasks to the NetworkManager.
///
/// `Broadcast` wraps a `SwarmMessage` for GossipSub. `SendTensor` and
/// `SendTensorResult` route tensor data through the Cap'n Proto
/// request_response protocol for zero-copy efficiency.
#[derive(Clone, Debug)]
pub enum NetworkCommand {
    /// Broadcast a message via GossipSub to all subscribers.
    Broadcast(SwarmMessage),
    /// Send a tensor forward pass to a specific peer via Cap'n Proto.
    SendTensor {
        target_peer_bytes: Vec<u8>,
        forward: LayerForward,
    },
    /// Send a tensor result back to a specific peer via Cap'n Proto.
    SendTensorResult {
        target_peer_bytes: Vec<u8>,
        result: LayerResult,
    },
    /// Send a shard transfer request to a specific peer.
    SendShardRequest {
        target_peer_bytes: Vec<u8>,
        request: ShardRequest,
    },
}

// ---- Rebalancing ----
/// Events that trigger shard rebalancing.
#[derive(Clone, Debug)]
pub enum RebalanceEvent {
    PeerJoined(NodeId),
    PeerLeft(NodeId),
    DiskPressure { available_mb: u64 },
    ManualTrigger,
}

// ---- Peer State ----
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
}

// ---- Node Stats ----
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeStats {
    pub peers_connected: u32,
    pub requests_served: u64,
    pub requests_made: u64,
    pub bytes_uploaded: u64,
    pub bytes_downloaded: u64,
    pub uptime_start: chrono::DateTime<chrono::Utc>,
    /// NAT status detected by AutoNAT ("Public", "Private", "Unknown").
    #[serde(default)]
    pub nat_status: Option<String>,
}

impl Default for NodeStats {
    fn default() -> Self {
        Self {
            peers_connected: 0,
            requests_served: 0,
            requests_made: 0,
            bytes_uploaded: 0,
            bytes_downloaded: 0,
            uptime_start: chrono::Utc::now(),
            nat_status: None,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_display_shows_short_hex() {
        let id = NodeId([0xab; 32]);
        assert_eq!(format!("{id}"), "abababababababab");
    }

    #[test]
    fn model_id_display() {
        let id = ModelId("llama3-70b-q4km".into());
        assert_eq!(format!("{id}"), "llama3-70b-q4km");
    }

    #[test]
    fn sampling_params_default() {
        let params = SamplingParams::default();
        assert!((params.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(params.max_tokens, 2048);
        assert_eq!(params.top_k, 40);
    }

    #[test]
    fn chat_message_serde_roundtrip() {
        let msg = ChatMessage {
            role: Role::User,
            content: "hello".into(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "hello");
    }

    #[test]
    fn role_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
    }

    #[test]
    fn swarm_message_serde_roundtrip() {
        let msg = SwarmMessage::HealthPing {
            nonce: 42,
            timestamp: 1000,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let parsed: SwarmMessage = serde_json::from_str(&json).unwrap();
        match parsed {
            SwarmMessage::HealthPing { nonce, timestamp } => {
                assert_eq!(nonce, 42);
                assert_eq!(timestamp, 1000);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn shard_id_equality() {
        let a = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        let b = ShardId {
            model_id: ModelId("test".into()),
            index: 0,
        };
        assert_eq!(a, b);
    }
}
