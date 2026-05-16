//! Credit-economy + device-pool configuration.
//!
//! Hosts the per-rate `CreditRateConfig` and the `PoolConfig` (max pool
//! size, invitation TTL, rate limits, private/offline mode flags). Pool
//! credit-rate overrides are nested via `PoolConfig::credit_rates`, so
//! the two types are co-located.

use super::default_true;
use serde::{Deserialize, Serialize};

/// Configurable credit earn/spend rates per pool or globally.
/// All values are in credits per unit of work.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditRateConfig {
    /// Credits earned per layer per token for serving inference.
    #[serde(default = "default_rate_inference_serve")]
    pub inference_serve: i64,
    /// Credits spent per layer per token for consuming inference.
    #[serde(default = "default_rate_inference_consume")]
    pub inference_consume: i64,
    /// Credits earned per GB per hour for hosting shards.
    #[serde(default = "default_rate_shard_hosting")]
    pub shard_hosting: i64,
    /// Credits earned per GB transferred for seeding shards.
    #[serde(default = "default_rate_shard_seeding")]
    pub shard_seeding: i64,
    /// Credits earned per connection hour for relay service.
    #[serde(default = "default_rate_relay_service")]
    pub relay_service: i64,
    /// Credits deducted as penalty for serve failures.
    #[serde(default = "default_rate_penalty")]
    pub penalty_serve_failure: i64,
}

impl Default for CreditRateConfig {
    fn default() -> Self {
        Self {
            inference_serve: default_rate_inference_serve(),
            inference_consume: default_rate_inference_consume(),
            shard_hosting: default_rate_shard_hosting(),
            shard_seeding: default_rate_shard_seeding(),
            relay_service: default_rate_relay_service(),
            penalty_serve_failure: default_rate_penalty(),
        }
    }
}

fn default_rate_inference_serve() -> i64 {
    10
}
fn default_rate_inference_consume() -> i64 {
    10
}
fn default_rate_shard_hosting() -> i64 {
    1
}
fn default_rate_shard_seeding() -> i64 {
    5
}
fn default_rate_relay_service() -> i64 {
    2
}
fn default_rate_penalty() -> i64 {
    50
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoolConfig {
    #[serde(default = "default_max_pool_size")]
    pub max_pool_size: u32,
    #[serde(default = "default_invitation_ttl_hours")]
    pub invitation_ttl_hours: u32,
    #[serde(default = "default_pool_rate_limit")]
    pub rate_limit_per_hour: u32,
    #[serde(default = "default_pool_gossip_interval")]
    pub gossip_interval_secs: u64,
    /// Global credit rate overrides. Pools can further override these per-pool.
    #[serde(default)]
    pub credit_rates: CreditRateConfig,
    /// Private mode: restrict all inference and shard management to pool members only.
    /// When enabled, no data leaves your device pool.
    #[serde(default)]
    pub private_mode: bool,
    /// When private mode is on, also allow LAN peers (discovered via mDNS) as inference targets.
    #[serde(default = "default_true")]
    pub private_mode_allow_lan: bool,
    /// Offline LAN mode: disable internet bootstrap, mDNS-only discovery. Air-gapped operation.
    #[serde(default)]
    pub offline_mode: bool,
    /// R134: opt-in pool-state diff gossip. When on, the pool owner emits
    /// `PoolMessage::StateDiff` between full broadcasts — added/removed
    /// members + a signed checksum — instead of always sending the full
    /// member list. Periodic broadcasts and the first broadcast after
    /// restart remain full-state to bound recovery time for late
    /// joiners. Off by default; flip on once a WAN bench shows the
    /// trailing-full-state broadcast is bandwidth-constrained for your
    /// pool size.
    #[serde(default)]
    pub state_diff_gossip: bool,
    /// R134: opt-in cross-pool model catalog gossip. When on, the pool
    /// owner periodically broadcasts the model IDs the pool can serve
    /// on the regions GossipSub topic. Outsiders cache this as a
    /// discovery signal — "Pool X also serves Y" — but cross-pool
    /// routing is NOT enabled by this flag (the private-mode contract
    /// is preserved). Requires the pool to have at least
    /// `share_model_catalog_min_members` members to actually publish —
    /// k-anonymity floor prevents the channel from being used to
    /// enumerate small private pools.
    #[serde(default)]
    pub share_model_catalog: bool,
    /// R134: k-anonymity floor for `share_model_catalog`. Pools smaller
    /// than this never publish their catalog regardless of the
    /// `share_model_catalog` flag.
    #[serde(default = "default_share_model_catalog_min")]
    pub share_model_catalog_min_members: u32,
    /// R134.7: opt-in cross-pool inference routing. When BOTH this
    /// flag AND `private_mode` are on, the scheduler may route
    /// inference for a model to a foreign pool's members IF that pool
    /// has advertised serving the model via `foreign_pool_catalog` AND
    /// no member of the local pool currently holds the model. Default
    /// off — preserves the existing "your inference stays in your pool"
    /// contract until the user explicitly opts in. Note: the catalog
    /// is opt-in to publish; routing is opt-in to consume. Both sides
    /// must agree for cross-pool requests to actually flow.
    #[serde(default)]
    pub allow_cross_pool_inference: bool,
}

fn default_share_model_catalog_min() -> u32 {
    3
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: default_max_pool_size(),
            invitation_ttl_hours: default_invitation_ttl_hours(),
            rate_limit_per_hour: default_pool_rate_limit(),
            gossip_interval_secs: default_pool_gossip_interval(),
            credit_rates: CreditRateConfig::default(),
            private_mode: false,
            private_mode_allow_lan: true,
            offline_mode: false,
            state_diff_gossip: false,
            share_model_catalog: false,
            share_model_catalog_min_members: default_share_model_catalog_min(),
            allow_cross_pool_inference: false,
        }
    }
}

fn default_max_pool_size() -> u32 {
    10
}

fn default_invitation_ttl_hours() -> u32 {
    24
}

fn default_pool_rate_limit() -> u32 {
    10
}

fn default_pool_gossip_interval() -> u64 {
    600
}
