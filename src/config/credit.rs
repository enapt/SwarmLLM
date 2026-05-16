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
