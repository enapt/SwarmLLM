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
    /// SEC-M15: Should be called periodically (every 5 minutes) from a background task
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
        self.subnet_counts.retain(|_, nodes| {
            nodes.retain(|(_, ts)| *ts > subnet_cutoff);
            !nodes.is_empty()
        });
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

    /// Register a node's observed IPv4 address for subnet clustering detection.
    /// Extracts the /24 prefix and tracks which NodeIds share it.
    pub fn register_subnet(&mut self, node_id: &NodeId, ip_bytes: [u8; 4]) {
        let prefix = [ip_bytes[0], ip_bytes[1], ip_bytes[2]];
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
