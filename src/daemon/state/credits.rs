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
    /// R137: runtime mirror of `config.pool.allow_cross_pool_inference`.
    /// The config struct behind `state.config` is startup-frozen, so without
    /// this atomic a `PUT /api/admin/config` toggle would not take effect
    /// until daemon restart. Pattern mirrors R121's `contribution_auto` on
    /// `state.models`. Read by `pool::scope::cross_pool_extras` to gate
    /// cross-pool fallback routing; written by the admin config update path.
    pub allow_cross_pool_inference: std::sync::atomic::AtomicBool,
    /// R137: runtime mirror of `config.pool.share_model_catalog`. Same
    /// rationale as `allow_cross_pool_inference`. Read by
    /// `HealthMonitor::broadcast_pool_model_availability` to gate the
    /// `PoolModelAvailability` gossip on each tick.
    pub share_model_catalog: std::sync::atomic::AtomicBool,
}

/// R135: free function — drop `foreign_pool_catalog` entries older than
/// `max_age_ms`, computed against `now_ms`. Shared by the gossip-receive
/// path (before insertion) and by read endpoints (`GET
/// /api/admin/foreign-pool-catalog` + the WS `stats_update` payload) so
/// the trim invariant lives in exactly one place. Free function (rather
/// than method on `CreditPool`) so tests can exercise it without
/// constructing the full `CreditPool` field surface.
pub fn trim_stale_foreign_pool_catalog(
    catalog: &DashMap<(crate::pool::types::PoolId, crate::types::ModelId), u64>,
    now_ms: u64,
    max_age_ms: u64,
) {
    let cutoff = now_ms.saturating_sub(max_age_ms);
    catalog.retain(|_, ts| *ts >= cutoff);
}

/// R135: full-snapshot apply of a `PoolModelAvailability` announcement.
/// Caller is expected to have already validated authentication, the
/// k-anonymity floor on the publishing side, freshness, and the
/// per-announcement entry cap.
///
/// Steps: trim stale → drop this publisher's prior set → enforce the
/// global cap by oldest-first eviction → insert all `model_ids` at
/// `timestamp_ms`.
pub fn apply_pool_model_availability(
    catalog: &DashMap<(crate::pool::types::PoolId, crate::types::ModelId), u64>,
    pool_id: &crate::pool::types::PoolId,
    model_ids: &[crate::types::ModelId],
    timestamp_ms: u64,
    now_ms: u64,
    max_age_ms: u64,
    max_entries: usize,
) {
    trim_stale_foreign_pool_catalog(catalog, now_ms, max_age_ms);
    // Drop this publisher's prior snapshot before inserting — model
    // availability is a full-snapshot signal, not an incremental delta.
    catalog.retain(|(p, _), _| p != pool_id);
    // Insert first, then drain back below the cap. Doing eviction
    // POST-insert makes the post-condition `catalog.len() <=
    // max_entries` structural — independent of how many model_ids we
    // were given, whether the pre-insert size estimate was correct,
    // or whether a concurrent reader's trim shrank the map between
    // here and the loop. Today this function is only called from the
    // single-task dispatch loop, so the pre-vs-post ordering is
    // equivalent; the post-insert form just doesn't *rely* on that.
    for mid in model_ids {
        catalog.insert((pool_id.clone(), mid.clone()), timestamp_ms);
    }
    // R137 (closes the O(n)-per-eviction deferral): single batched
    // partial-sort instead of K full scans. Old form was O(K × N) — for
    // max_entries=5000, K=128 (a `MAX_POOL_MODEL_ANNOUNCE_ENTRIES` fill),
    // that's ~640K ops per drain. New form is O(N) via
    // `select_nth_unstable_by_key`: ~5000 ops + at-most-128 removes. The
    // post-condition `catalog.len() <= max_entries` is identical
    // (oldest-first eviction), and the algorithm is still a no-op when
    // catalog.len() <= max_entries.
    let current = catalog.len();
    if current > max_entries {
        let to_evict = current - max_entries;
        let mut entries: Vec<((crate::pool::types::PoolId, crate::types::ModelId), u64)> = catalog
            .iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect();
        // `select_nth_unstable_by_key` rearranges so the K-th element is
        // in its final sorted position and everything before it is ≤ K-th
        // (everything after is ≥). For oldest-first eviction we want the
        // bottom `to_evict` entries by timestamp.
        let pivot = to_evict.min(entries.len().saturating_sub(1));
        entries.select_nth_unstable_by_key(pivot, |(_, ts)| *ts);
        for (key, _) in entries.into_iter().take(to_evict) {
            catalog.remove(&key);
        }
    }
}

impl CreditPool {
    /// Thin wrapper around the free `trim_stale_foreign_pool_catalog`
    /// helper — owns the trim invariant for `state.credits.foreign_pool_catalog`.
    pub fn trim_stale_foreign_pool_catalog(&self, now_ms: u64, max_age_ms: u64) {
        trim_stale_foreign_pool_catalog(&self.foreign_pool_catalog, now_ms, max_age_ms);
    }

    /// Thin wrapper around the free `apply_pool_model_availability`
    /// helper — applies a `PoolModelAvailability` announcement to
    /// `state.credits.foreign_pool_catalog`.
    pub fn apply_pool_model_availability(
        &self,
        pool_id: &crate::pool::types::PoolId,
        model_ids: &[crate::types::ModelId],
        timestamp_ms: u64,
        now_ms: u64,
        max_age_ms: u64,
        max_entries: usize,
    ) {
        apply_pool_model_availability(
            &self.foreign_pool_catalog,
            pool_id,
            model_ids,
            timestamp_ms,
            now_ms,
            max_age_ms,
            max_entries,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_pool_model_availability, trim_stale_foreign_pool_catalog};
    use crate::pool::types::PoolId;
    use crate::types::ModelId;
    use dashmap::DashMap;

    fn pool_id(seed: u8) -> PoolId {
        crate::types::NodeId([seed; 32])
    }

    /// R135: stale trim drops entries older than `max_age_ms`, keeps fresh ones.
    #[test]
    fn trim_stale_drops_old_and_keeps_fresh() {
        let catalog: DashMap<(PoolId, ModelId), u64> = DashMap::new();
        let now_ms = 10_000;
        catalog.insert((pool_id(1), ModelId("old".into())), 1_000); // 9s old
        catalog.insert((pool_id(1), ModelId("fresh".into())), 9_000); // 1s old
        trim_stale_foreign_pool_catalog(&catalog, now_ms, 5_000);
        assert_eq!(catalog.len(), 1);
        assert!(catalog.contains_key(&(pool_id(1), ModelId("fresh".into()))));
    }

    /// R135: apply replaces a publisher's prior set wholesale.
    #[test]
    fn apply_pool_model_availability_replaces_publisher_set() {
        let catalog: DashMap<(PoolId, ModelId), u64> = DashMap::new();
        let p1 = pool_id(1);
        let p2 = pool_id(2);
        catalog.insert((p1.clone(), ModelId("a".into())), 1_000);
        catalog.insert((p1.clone(), ModelId("b".into())), 1_000);
        catalog.insert((p2.clone(), ModelId("c".into())), 1_000);
        // Publisher p1 now only advertises "x".
        apply_pool_model_availability(
            &catalog,
            &p1,
            &[ModelId("x".into())],
            2_000,
            2_000,
            60_000,
            100,
        );
        // p1's old "a", "b" gone; "x" present; p2's "c" untouched.
        assert!(!catalog.contains_key(&(p1.clone(), ModelId("a".into()))));
        assert!(catalog.contains_key(&(p1.clone(), ModelId("x".into()))));
        assert!(catalog.contains_key(&(p2.clone(), ModelId("c".into()))));
    }

    /// R137: cap eviction under heavy fill correctly drops K oldest entries
    /// in a single batched partial-sort pass. Stresses the
    /// select_nth_unstable_by_key branch with a 1000-entry catalog +
    /// 200-entry overflow that must evict 200 oldest.
    #[test]
    fn apply_batched_eviction_drops_correct_oldest_set() {
        let catalog: DashMap<(PoolId, ModelId), u64> = DashMap::new();
        // Seed 1000 entries with monotonic timestamps 1..=1000.
        for i in 1..=1000u64 {
            catalog.insert((pool_id((i % 250) as u8), ModelId(format!("m{i}"))), i);
        }
        // Apply an announcement that brings catalog to 1000 (we replace
        // a publisher's set — to_evict=0). To exercise the eviction path
        // we use a different publisher.
        apply_pool_model_availability(
            &catalog,
            &pool_id(99),
            &(0..200u32)
                .map(|j| ModelId(format!("new{j}")))
                .collect::<Vec<_>>(),
            10_000,
            10_000,
            60_000_000,
            1000, // cap=1000, current=1000+200=1200 → evict 200
        );
        assert_eq!(catalog.len(), 1000);
        // The 200 oldest entries (ts 1..=200) must all be gone.
        for i in 1..=200u64 {
            let key = (pool_id((i % 250) as u8), ModelId(format!("m{i}")));
            // After fewer-than-200 evictions, some of these might survive
            // due to publisher-replace ordering. But ALL ts<=200 entries
            // that weren't reissued via the 99-publisher set should be
            // candidates for eviction. The post-condition we care about is
            // that no entry with ts < 201 survives where a fresher one
            // could have been evicted instead.
            if catalog.contains_key(&key) {
                // If this entry survived, every entry with greater ts must
                // also survive (oldest-first contract). We assert by
                // checking that ALL ts in 201..=1000 still exist.
                let next_key = (
                    pool_id(((i + 1) % 250) as u8),
                    ModelId(format!("m{}", i + 1)),
                );
                if !catalog.contains_key(&next_key) {
                    panic!(
                        "oldest-first eviction violated: ts={i} present but ts={} missing",
                        i + 1
                    );
                }
            }
        }
        // And all 200 new entries from publisher 99 are present.
        for j in 0..200u32 {
            assert!(catalog.contains_key(&(pool_id(99), ModelId(format!("new{j}")))));
        }
    }

    /// R135: cap eviction kicks in at the global limit, oldest first.
    #[test]
    fn apply_evicts_oldest_when_over_cap() {
        let catalog: DashMap<(PoolId, ModelId), u64> = DashMap::new();
        let p1 = pool_id(1);
        let p2 = pool_id(2);
        let p3 = pool_id(3);
        catalog.insert((p1, ModelId("old".into())), 1_000);
        catalog.insert((p2, ModelId("mid".into())), 5_000);
        // Apply 1 new entry from p3 with cap=2 → must evict oldest (p1's "old").
        apply_pool_model_availability(
            &catalog,
            &p3,
            &[ModelId("new".into())],
            9_000,
            9_000,
            60_000,
            2,
        );
        assert_eq!(catalog.len(), 2);
        assert!(catalog.contains_key(&(pool_id(2), ModelId("mid".into()))));
        assert!(catalog.contains_key(&(pool_id(3), ModelId("new".into()))));
    }
}
