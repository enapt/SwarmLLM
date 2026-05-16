use std::sync::Arc;

use dashmap::DashMap;
use tokio::sync::{mpsc, RwLock};

use crate::types::{CreditBalance, NodeId};

/// Credit & pool: balances, pool membership, escrow, trust, anti-gaming.
pub struct CreditPool {
    pub credit_balance: Arc<RwLock<CreditBalance>>,
    pub pending_credit_earn: std::sync::atomic::AtomicI64,
    pub pool_state: RwLock<Option<crate::pool::types::PoolState>>,
    pub pool_registry: DashMap<crate::pool::types::PoolId, crate::pool::types::PoolState>,
    pub pool_tx: RwLock<Option<mpsc::Sender<crate::pool::types::PoolCommand>>>,
    pub pool_credit_rates: DashMap<NodeId, crate::config::CreditRateConfig>,
    pub trust_manager: crate::credit::trust::TrustManager,
    pub escrow_manager: Arc<crate::credit::escrow::EscrowManager>,
    pub anti_gaming: tokio::sync::Mutex<crate::credit::anti_gaming::AntiGaming>,
    pub peer_credit_balances: DashMap<NodeId, i64>,
    /// Cached (computed_at, percentile) to avoid O(n) scan of peer_credit_balances on
    /// every inference submission. Staleness of a few hundred ms is fine — the result
    /// is only used to pick a quantized priority tier (Bronze/Silver/Gold/Platinum).
    pub credit_percentile_cache: parking_lot::Mutex<(std::time::Instant, f32)>,
    /// Private mode: restrict inference + auto-manage to pool members (+ optional LAN peers).
    pub private_mode: std::sync::atomic::AtomicBool,
    /// Offline mode: no internet bootstrap, mDNS-only, no automatic HF downloads.
    pub offline_mode: std::sync::atomic::AtomicBool,
    /// R134: discovery cache for inter-pool model availability announcements.
    /// Keyed by `(announcing_pool_id, model_id)`; value is `(received_at_ms)`.
    /// Trimmed on every read against `FOREIGN_POOL_CATALOG_MAX_AGE_MS`. Cap
    /// `MAX_FOREIGN_POOL_CATALOG_ENTRIES` is enforced on insertion. This is a
    /// *discovery* surface only — does NOT change routing decisions; the
    /// private-mode contract is preserved.
    pub foreign_pool_catalog: DashMap<(crate::pool::types::PoolId, crate::types::ModelId), u64>,
}
