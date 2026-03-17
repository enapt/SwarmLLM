use crate::types::PriorityTier;

/// Calculate the priority tier for a node based on its balance and network percentile.
///
/// Tier thresholds (from the spec):
/// - Platinum: >= 90th percentile
/// - Gold: >= 70th percentile
/// - Silver: positive balance
/// - Bronze: zero or negative balance
pub fn calculate_tier(balance: i64, network_percentile: f32) -> PriorityTier {
    let tier = if network_percentile >= 0.90 {
        PriorityTier::Platinum
    } else if network_percentile >= 0.70 {
        PriorityTier::Gold
    } else if balance > 0 {
        PriorityTier::Silver
    } else {
        PriorityTier::Bronze
    };
    tracing::debug!(
        balance,
        network_percentile,
        tier = ?tier,
        "DIAG: calculate_tier"
    );
    tier
}

/// Utility for tier name resolution (used by admin API).
/// SEC-I2: Delegates to `calculate_tier()` to avoid inconsistent tier logic.
pub struct PriorityCalculator;

impl PriorityCalculator {
    /// Return a human-readable tier name based on the tier enum.
    /// When percentile data is unavailable, uses a default of 0.5.
    pub fn tier_name(balance: i64) -> &'static str {
        // Delegate to the canonical tier calculation with a default percentile
        let tier = calculate_tier(balance, if balance > 0 { 0.5 } else { 0.0 });
        match tier {
            PriorityTier::Platinum => "platinum",
            PriorityTier::Gold => "gold",
            PriorityTier::Silver => "silver",
            PriorityTier::Bronze => "bronze",
        }
    }
}

/// Calculate the maximum concurrent requests allowed for a tier.
pub fn max_concurrent_for_tier(tier: PriorityTier, base_max: usize) -> usize {
    match tier {
        PriorityTier::Bronze => (base_max / 4).max(1),
        PriorityTier::Silver => (base_max / 2).max(1),
        PriorityTier::Gold => base_max,
        PriorityTier::Platinum => base_max * 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_calculation_platinum() {
        assert_eq!(calculate_tier(10000, 0.95), PriorityTier::Platinum);
        assert_eq!(calculate_tier(100, 0.90), PriorityTier::Platinum);
    }

    #[test]
    fn tier_calculation_gold() {
        assert_eq!(calculate_tier(5000, 0.75), PriorityTier::Gold);
        assert_eq!(calculate_tier(100, 0.70), PriorityTier::Gold);
    }

    #[test]
    fn tier_calculation_silver() {
        assert_eq!(calculate_tier(1, 0.5), PriorityTier::Silver);
        assert_eq!(calculate_tier(100, 0.3), PriorityTier::Silver);
    }

    #[test]
    fn tier_calculation_bronze() {
        assert_eq!(calculate_tier(0, 0.5), PriorityTier::Bronze);
        assert_eq!(calculate_tier(-100, 0.3), PriorityTier::Bronze);
    }

    #[test]
    fn tier_ordering() {
        assert!(PriorityTier::Platinum > PriorityTier::Gold);
        assert!(PriorityTier::Gold > PriorityTier::Silver);
        assert!(PriorityTier::Silver > PriorityTier::Bronze);
    }

    #[test]
    fn tier_weights_increase() {
        fn tier_weight(tier: PriorityTier) -> u32 {
            match tier {
                PriorityTier::Bronze => 1,
                PriorityTier::Silver => 2,
                PriorityTier::Gold => 4,
                PriorityTier::Platinum => 8,
            }
        }
        assert!(tier_weight(PriorityTier::Platinum) > tier_weight(PriorityTier::Gold));
        assert!(tier_weight(PriorityTier::Gold) > tier_weight(PriorityTier::Silver));
        assert!(tier_weight(PriorityTier::Silver) > tier_weight(PriorityTier::Bronze));
    }

    #[test]
    fn concurrent_limits_by_tier() {
        let base = 8;
        assert_eq!(max_concurrent_for_tier(PriorityTier::Bronze, base), 2);
        assert_eq!(max_concurrent_for_tier(PriorityTier::Silver, base), 4);
        assert_eq!(max_concurrent_for_tier(PriorityTier::Gold, base), 8);
        assert_eq!(max_concurrent_for_tier(PriorityTier::Platinum, base), 16);
    }
}
