use crate::types::PriorityTier;

/// The tier every request gets while the credit economy is dormant.
///
/// Silver rather than Gold deliberately: it keeps a per-requester concurrency
/// cap (½ of `max_concurrent_requests`) so a single peer still cannot take the
/// whole queue, which is the one thing the tier system was genuinely providing.
/// Gold or Platinum would remove that isolation as a side effect of removing
/// the economy.
pub const DORMANT_TIER: PriorityTier = PriorityTier::Silver;

/// The priority tier for a node. **Currently the same for everyone.**
///
/// This used to read the balance and a gossiped network percentile:
/// Platinum ≥ 90th percentile, Gold ≥ 70th, Silver for any positive balance,
/// Bronze otherwise. Combined with [`max_concurrent_for_tier`] that gave a
/// high-balance node up to 8× the concurrency of a low-balance one.
///
/// It is flat because the balance driving it is **self-minted**: no credit has
/// ever moved between two nodes, and nothing stopped a node inflating its own
/// figure by serving itself (`docs/CREDITS_DESIGN.md` § 1). So the tier was not
/// measuring contribution, it was measuring how much a node had done *for
/// itself* — and then handing out real throughput for it.
///
/// The arguments are kept rather than deleted. They are what the real
/// implementation consults, the call sites already compute them correctly, and
/// removing them would mean reconstructing that plumbing later; the
/// `_`-prefixes make the current behaviour impossible to misread as a bug.
/// Restoring the mapping means restoring this body — and satisfying
/// `docs/CREDITS_DESIGN.md` § 6 first.
pub fn calculate_tier(_balance: i64, _network_percentile: f32) -> PriorityTier {
    DORMANT_TIER
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

    /// While the economy is dormant, a balance must buy nothing.
    ///
    /// These inputs are the ones that used to span the whole range — the top of
    /// Platinum down to a deeply negative Bronze. All four now land on the same
    /// tier, which is the property that matters: a node that has minted itself
    /// a large number gets exactly the service a node with none does.
    #[test]
    fn a_self_minted_balance_buys_no_priority() {
        let across_the_old_range = [
            (10_000_i64, 0.95_f32), // was Platinum
            (5_000, 0.75),          // was Gold
            (1, 0.5),               // was Silver
            (0, 0.5),               // was Bronze
            (-100, 0.3),            // was Bronze
            (-10_000, 0.0),         // was Bronze
        ];
        for (balance, percentile) in across_the_old_range {
            assert_eq!(
                calculate_tier(balance, percentile),
                DORMANT_TIER,
                "balance {balance} / percentile {percentile} changed the tier — \
                 credits are dormant and must not affect service \
                 (docs/CREDITS_DESIGN.md)"
            );
        }
    }

    /// The dormant tier must still cap one requester below the whole queue.
    ///
    /// Flattening the tiers removed the credit advantage; it must not also
    /// remove the per-requester isolation, which is the one thing the tier
    /// system was really providing. Gold or Platinum would have done exactly
    /// that as a side effect.
    #[test]
    fn the_dormant_tier_still_isolates_one_requester() {
        let base = 8;
        let cap = max_concurrent_for_tier(DORMANT_TIER, base);
        assert!(
            cap < base,
            "the dormant tier ({DORMANT_TIER:?}) lets one requester take the \
             entire queue ({cap} of {base}) — pick a tier that still caps"
        );
        assert!(
            cap >= 1,
            "the cap must leave a requester able to make progress"
        );
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
