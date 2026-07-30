use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::NodeId;

/// Maximum number of nodes per /24 subnet before triggering elevated scrutiny.
const SUBNET_CLUSTER_THRESHOLD: usize = 5;

/// Elevated spot-check rate for nodes in clustered subnets.
const SUBNET_CLUSTER_SPOT_CHECK_RATE: f64 = 0.25;

/// Age (seconds) after which a subnet registration is evicted from `subnet_counts`.
/// Prevents unbounded growth from nodes that connect but never transact. 1 hour.
const SUBNET_EVICTION_SECS: u64 = 3_600;

/// Default rate-limit window (seconds) used by `AntiGaming::new()` — caps how
/// recent transactions are counted toward the per-peer rate limit.
const ANTI_GAMING_RATE_WINDOW_SECS: u64 = 300;

/// Peer-count threshold above which `cleanup` emits a warning that the
/// rate-limit table is unusually large. Operators may want to alert on this.
const RATE_LIMITS_WARN_THRESHOLD: usize = 10_000;

/// Rate limiter and anti-gaming checks for the credit system.
///
/// Prevents:
/// - Rapid-fire transactions from a single peer (rate limiting)
/// - Self-dealing (same node as from/to)
/// - Transaction amounts that are implausibly large
/// - Spot-check verification of claimed work
pub struct AntiGaming {
    /// Per-peer rate limiting: tracks last N transaction timestamps.
    rate_limits: HashMap<NodeId, Vec<Instant>>,
    /// Maximum transactions per peer per window.
    max_tx_per_window: usize,
    /// Rate limit window duration.
    window_duration: Duration,
    /// Maximum single transaction amount.
    max_transaction_amount: i64,
    /// Spot-check probability (0.0 to 1.0).
    spot_check_rate: f64,
    /// Tracks observed /24 subnet clustering (first 3 bytes of IPv4).
    /// Many nodes from the same /24 may indicate Sybil attack.
    /// Each entry stores (NodeId, registration time) for age-based eviction.
    subnet_counts: HashMap<[u8; 3], Vec<(NodeId, Instant)>>,
    /// Reverse index: NodeId → its current /24 prefix. Lets `register_subnet`
    /// remove a node from its old bucket in O(1) instead of scanning every
    /// bucket. Without this, `register_subnet` was O(N × buckets) and ran
    /// on every libp2p Identify event — Sybil attackers opening many
    /// distinct connections forced quadratic work per push.
    node_subnet: HashMap<NodeId, [u8; 3]>,
}

impl AntiGaming {
    pub fn new() -> Self {
        Self {
            rate_limits: HashMap::new(),
            max_tx_per_window: 100,
            window_duration: Duration::from_secs(ANTI_GAMING_RATE_WINDOW_SECS),
            max_transaction_amount: 100_000,
            spot_check_rate: 0.05, // 5% of transactions
            subnet_counts: HashMap::new(),
            node_subnet: HashMap::new(),
        }
    }

    /// Check whether a transaction should be allowed based on rate limits and validity.
    fn check_transaction(
        &mut self,
        from: &NodeId,
        to: &NodeId,
        amount: i64,
    ) -> Result<SpotCheckDecision, AntiGamingViolation> {
        // Check self-dealing
        if from == to {
            tracing::debug!(from = %from, "DIAG: anti_gaming violation — self dealing");
            return Err(AntiGamingViolation::SelfDealing);
        }

        // Check amount bounds
        if amount <= 0 {
            return Err(AntiGamingViolation::InvalidAmount(amount));
        }
        if amount > self.max_transaction_amount {
            return Err(AntiGamingViolation::ExcessiveAmount {
                amount,
                max: self.max_transaction_amount,
            });
        }

        // Rate limit check
        if !self.check_rate_limit(from) {
            return Err(AntiGamingViolation::RateLimited {
                node: from.clone(),
                window_secs: self.window_duration.as_secs(),
            });
        }

        // Decide whether to spot-check (elevated rate for clustered subnets)
        let effective_rate = self.effective_spot_check_rate(from);
        let should_spot_check = rand::random::<f64>() < effective_rate;

        Ok(if should_spot_check {
            SpotCheckDecision::RequiresVerification
        } else {
            SpotCheckDecision::Approved
        })
    }

    /// Record a transaction for rate limiting purposes.
    fn record_transaction(&mut self, node: &NodeId) {
        let entries = self.rate_limits.entry(node.clone()).or_default();
        entries.push(Instant::now());
    }

    /// SEC-C4 + SEC-M21: Atomic check-and-record — validates transaction and records it
    /// in a single call to prevent TOCTOU races.
    pub fn check_and_record_transaction(
        &mut self,
        from: &NodeId,
        to: &NodeId,
        amount: i64,
    ) -> Result<SpotCheckDecision, AntiGamingViolation> {
        let decision = self.check_transaction(from, to, amount)?;
        self.record_transaction(from);
        Ok(decision)
    }

    /// Cleanup expired rate limit entries.
    /// SEC-M15: Should be called periodically (every health-monitor tick)
    /// to prevent unbounded memory growth in the rate_limits HashMap.
    pub fn cleanup(&mut self) {
        let cutoff = Instant::now() - self.window_duration;
        self.rate_limits.retain(|_, timestamps| {
            timestamps.retain(|t| *t > cutoff);
            !timestamps.is_empty()
        });
        // Evict subnet registrations older than SUBNET_EVICTION_SECS to prevent
        // unbounded growth from nodes that connect but never transact.
        let subnet_cutoff = Instant::now() - Duration::from_secs(SUBNET_EVICTION_SECS);
        let mut evicted: Vec<NodeId> = Vec::new();
        self.subnet_counts.retain(|_, nodes| {
            nodes.retain(|(n, ts)| {
                let keep = *ts > subnet_cutoff;
                if !keep {
                    evicted.push(n.clone());
                }
                keep
            });
            !nodes.is_empty()
        });
        // Keep `node_subnet` reverse index in sync — drop entries that were
        // just evicted from `subnet_counts` so a future register_subnet call
        // doesn't try to remove them from a bucket that no longer exists.
        for n in evicted {
            self.node_subnet.remove(&n);
        }
        // Soft warning if the rate-limit map grows unusually large between
        // ticks. Time-based eviction above bounds this in steady state, but
        // a Sybil burst could push it temporarily high. Emitted at ~once per
        // tick when the threshold is crossed.
        if self.rate_limits.len() > RATE_LIMITS_WARN_THRESHOLD {
            tracing::warn!(
                size = self.rate_limits.len(),
                threshold = RATE_LIMITS_WARN_THRESHOLD,
                "anti_gaming.rate_limits map exceeded soft threshold — possible burst of unique nodes"
            );
        }
    }

    /// Report a spot-check failure — peer claimed work they didn't do.
    pub fn report_spot_check_failure(&mut self, _node: &NodeId) -> PenaltyAction {
        PenaltyAction::ReduceTrust { amount: 0.1 }
    }

    /// Check rate limit for a node. Returns true if within limits.
    fn check_rate_limit(&mut self, node: &NodeId) -> bool {
        let cutoff = Instant::now() - self.window_duration;

        let entries = self.rate_limits.entry(node.clone()).or_default();
        entries.retain(|t| *t > cutoff);

        entries.len() < self.max_tx_per_window
    }

    /// /24 prefixes belonging to the project's own bootstrap anchors.
    ///
    /// Derived from `network::default_bootstrap_peers()` so adding an anchor
    /// allow-lists it automatically and the two lists cannot drift.
    ///
    /// **Why this exists.** The shipped anchor and its co-located infrastructure
    /// share a /24, so every NAT-bound node that talks to it saw
    /// `Subnet clustering detected subnet="212.132.104.0/24" node_count=6` —
    /// 17 times in 4.5 hours on one reporting node. It is a false positive
    /// against the project's own relay, it raised the spot-check rate against
    /// the one peer every relay-bound node depends on, and at WARN every few
    /// minutes it buried real warnings. Reported 2026-07-30.
    ///
    /// **This is a deliberate, narrow trust delegation**, the same shape as the
    /// trusted-publisher allowlist: a genuine Sybil farm co-located with the
    /// anchor would escape this heuristic. That is accepted because the anchor is
    /// project-operated, and because the heuristic only ever raised a
    /// spot-check rate — it never blocked anything.
    fn anchor_subnets() -> Vec<[u8; 3]> {
        crate::config::default_bootstrap_peers()
            .iter()
            .filter_map(|addr| {
                // "/ip4/212.132.104.177/tcp/..." → the octets after `/ip4/`.
                let rest = addr.strip_prefix("/ip4/")?;
                let ip = rest.split('/').next()?;
                let parts: Vec<u8> = ip.split('.').filter_map(|o| o.parse().ok()).collect();
                (parts.len() == 4).then(|| [parts[0], parts[1], parts[2]])
            })
            .collect()
    }

    /// Register a node's observed IPv4 address for subnet clustering detection.
    /// Extracts the /24 prefix and tracks which NodeIds share it.
    ///
    /// A node may move between subnets (NAT change, mobile/cellular handoff,
    /// VPN reconnect). Without removing the stale `(NodeId, _)` entry from
    /// the previous /24's bucket, the same NodeId would accumulate in
    /// multiple buckets and `is_subnet_clustered` would return true if ANY
    /// of those happens to be crowded — penalising a peer for a subnet they
    /// no longer belong to. Drop the stale entry first.
    pub fn register_subnet(&mut self, node_id: &NodeId, ip_bytes: [u8; 4]) {
        let prefix = [ip_bytes[0], ip_bytes[1], ip_bytes[2]];
        // The project's own anchors share a /24 with their co-located
        // infrastructure; counting them as clustering is a false positive
        // against the relay every NAT-bound node depends on.
        if Self::anchor_subnets().contains(&prefix) {
            return;
        }
        // O(1) reverse-index lookup of the previous bucket. If the node was
        // last seen in a different /24, drop it from that bucket; otherwise
        // we're refreshing in place. The previous full-scan implementation
        // was O(N × buckets) and ran on every libp2p Identify event.
        if let Some(&old_prefix) = self.node_subnet.get(node_id) {
            if old_prefix != prefix {
                if let Some(nodes) = self.subnet_counts.get_mut(&old_prefix) {
                    nodes.retain(|(n, _)| n != node_id);
                    if nodes.is_empty() {
                        self.subnet_counts.remove(&old_prefix);
                    }
                }
            }
        }
        self.node_subnet.insert(node_id.clone(), prefix);

        let nodes = self.subnet_counts.entry(prefix).or_default();
        if let Some(entry) = nodes.iter_mut().find(|(n, _)| n == node_id) {
            entry.1 = Instant::now(); // refresh timestamp
        } else {
            nodes.push((node_id.clone(), Instant::now()));
            if nodes.len() > SUBNET_CLUSTER_THRESHOLD {
                tracing::warn!(
                    subnet = format!("{}.{}.{}.0/24", prefix[0], prefix[1], prefix[2]),
                    node_count = nodes.len(),
                    "Subnet clustering detected — elevated spot-check rate"
                );
            }
        }
    }

    /// Check if a node is in a clustered subnet (> SUBNET_CLUSTER_THRESHOLD nodes
    /// sharing the same /24). Returns true if the node should face elevated scrutiny.
    fn is_subnet_clustered(&self, node_id: &NodeId) -> bool {
        self.subnet_counts.values().any(|nodes| {
            nodes.len() > SUBNET_CLUSTER_THRESHOLD && nodes.iter().any(|(n, _)| n == node_id)
        })
    }

    /// Get the effective spot-check rate for a node, considering subnet clustering.
    pub fn effective_spot_check_rate(&self, node_id: &NodeId) -> f64 {
        if self.is_subnet_clustered(node_id) {
            SUBNET_CLUSTER_SPOT_CHECK_RATE
        } else {
            self.spot_check_rate
        }
    }
}

impl Default for AntiGaming {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of an anti-gaming check.
#[derive(Debug, Clone, PartialEq)]
pub enum SpotCheckDecision {
    /// Transaction approved, no spot-check needed.
    Approved,
    /// Transaction approved, but requires spot-check verification.
    RequiresVerification,
}

/// A violation detected by the anti-gaming system.
#[derive(Debug, Clone)]
pub enum AntiGamingViolation {
    /// Node tried to transact with itself.
    SelfDealing,
    /// Transaction amount is zero or negative.
    InvalidAmount(i64),
    /// Transaction amount exceeds maximum.
    ExcessiveAmount { amount: i64, max: i64 },
    /// Node exceeded transaction rate limit.
    RateLimited { node: NodeId, window_secs: u64 },
}

impl std::fmt::Display for AntiGamingViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SelfDealing => write!(f, "Self-dealing detected"),
            Self::InvalidAmount(a) => write!(f, "Invalid transaction amount: {a}"),
            Self::ExcessiveAmount { amount, max } => {
                write!(f, "Excessive amount: {amount} (max: {max})")
            }
            Self::RateLimited { node, window_secs } => {
                write!(f, "Rate limited: {node} in {window_secs}s window")
            }
        }
    }
}

/// Action to take when a penalty is assessed.
#[derive(Debug, Clone)]
pub enum PenaltyAction {
    /// Reduce the peer's trust score.
    ReduceTrust { amount: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    #[test]
    fn rejects_self_dealing() {
        let mut ag = AntiGaming::new();
        let n = node(1);
        assert!(matches!(
            ag.check_transaction(&n, &n, 10),
            Err(AntiGamingViolation::SelfDealing)
        ));
    }

    #[test]
    fn rejects_zero_amount() {
        let mut ag = AntiGaming::new();
        assert!(matches!(
            ag.check_transaction(&node(1), &node(2), 0),
            Err(AntiGamingViolation::InvalidAmount(0))
        ));
    }

    #[test]
    fn rejects_negative_amount() {
        let mut ag = AntiGaming::new();
        assert!(matches!(
            ag.check_transaction(&node(1), &node(2), -5),
            Err(AntiGamingViolation::InvalidAmount(-5))
        ));
    }

    #[test]
    fn rejects_excessive_amount() {
        let mut ag = AntiGaming::new();
        assert!(matches!(
            ag.check_transaction(&node(1), &node(2), 200_000),
            Err(AntiGamingViolation::ExcessiveAmount { .. })
        ));
    }

    #[test]
    fn approves_valid_transaction() {
        let mut ag = AntiGaming::new();
        let result = ag.check_transaction(&node(1), &node(2), 100);
        assert!(result.is_ok());
    }

    #[test]
    fn rate_limits_after_threshold() {
        let mut ag = AntiGaming::new();
        ag.max_tx_per_window = 3;

        let from = node(1);
        let to = node(2);

        // First 3 should succeed
        for _ in 0..3 {
            assert!(ag.check_transaction(&from, &to, 10).is_ok());
            ag.record_transaction(&from);
        }

        // 4th should be rate limited
        assert!(matches!(
            ag.check_transaction(&from, &to, 10),
            Err(AntiGamingViolation::RateLimited { .. })
        ));
    }

    #[test]
    fn cleanup_removes_old_entries() {
        let mut ag = AntiGaming::new();
        // window=10ms, sleep=100ms — 10x margin so CI scheduler jitter
        // doesn't flake the test. Previous values (1ms+5ms) had only 4ms
        // headroom which was insufficient under shared-runner load.
        ag.window_duration = Duration::from_millis(10);
        ag.max_tx_per_window = 1;

        let from = node(1);
        let to = node(2);

        ag.record_transaction(&from);

        // Wait for window to expire
        std::thread::sleep(Duration::from_millis(100));
        ag.cleanup();

        // Should be allowed again
        assert!(ag.check_transaction(&from, &to, 10).is_ok());
    }

    #[test]
    fn spot_check_failure_returns_penalty() {
        let mut ag = AntiGaming::new();
        let action = ag.report_spot_check_failure(&node(1));
        match action {
            PenaltyAction::ReduceTrust { amount } => {
                assert!((amount - 0.1).abs() < f64::EPSILON);
            }
        }
    }
}

#[cfg(test)]
mod anchor_subnet_tests {
    use super::*;

    /// The shipped anchor's /24 must be recognised, so the project's own relay
    /// is not reported as a Sybil cluster to every NAT-bound node.
    #[test]
    fn the_shipped_anchor_subnet_is_recognised() {
        let subnets = AntiGaming::anchor_subnets();
        assert!(
            !subnets.is_empty(),
            "the bootstrap list should yield at least one /ip4/ anchor"
        );
        // Derived from the same list the daemon dials, so it cannot drift.
        for addr in crate::config::default_bootstrap_peers() {
            if let Some(rest) = addr.strip_prefix("/ip4/") {
                let ip = rest.split('/').next().unwrap();
                let o: Vec<u8> = ip.split('.').filter_map(|p| p.parse().ok()).collect();
                if o.len() == 4 {
                    assert!(
                        subnets.contains(&[o[0], o[1], o[2]]),
                        "anchor {ip} must be allow-listed"
                    );
                }
            }
        }
    }

    /// An unrelated subnet must still be tracked — this allowlist is narrow, not
    /// a hole in the heuristic.
    #[test]
    fn an_unrelated_subnet_is_still_tracked() {
        let subnets = AntiGaming::anchor_subnets();
        assert!(
            !subnets.contains(&[203, 0, 113]),
            "TEST-NET-3 is not an anchor"
        );
    }
}
