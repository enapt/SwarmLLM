//! Hourly performance rollups, persisted so a trend survives a restart.
//!
//! Deliberately NOT a time-series database. `monitoring/` already ships
//! Prometheus + Grafana and that is the trend store for anyone who wants one;
//! this exists for the far more common case of a user with no scrape target who
//! still wants to answer "is this release slower than the last one". One bucket
//! per hour, capped at a week, a few dozen bytes each.
//!
//! Only aggregates are kept. Per-request detail lives in the in-memory ring
//! (`recent_traces`) and is intentionally lost on restart — retaining
//! per-request, per-peer rows on disk is the unbounded-growth trap that
//! `docs/FUTURE_WORK.md` § Observability warns against.

use std::collections::HashMap;

/// Buckets retained: one week at hourly granularity.
pub const MAX_HOURLY_BUCKETS: usize = 168;

/// redb key for the persisted rollup series.
const DB_KEY: &str = "perf_hourly";

/// One hour of request outcomes.
///
/// Sums rather than averages, so buckets can be merged and re-averaged without
/// the error that averaging averages introduces.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct HourBucket {
    /// Unix ms at the start of the hour this bucket covers.
    pub hour_start_ms: u64,
    pub requests: u32,
    pub errors: u32,
    /// Sum of end-to-end durations, ms.
    pub total_ms_sum: u64,
    /// Sum of TTFT, ms, over the requests that HAD a first-token stamp — which
    /// is not all of them, hence the separate count.
    pub ttft_ms_sum: u64,
    pub ttft_count: u32,
    pub completion_tokens: u64,
    /// Requests per route name. At most 5 entries.
    pub by_route: HashMap<String, u32>,
}

impl HourBucket {
    /// Mean end-to-end duration, or `None` for an empty bucket.
    pub fn avg_total_ms(&self) -> Option<f64> {
        (self.requests > 0).then(|| self.total_ms_sum as f64 / self.requests as f64)
    }

    /// Mean TTFT over requests that reported one.
    pub fn avg_ttft_ms(&self) -> Option<f64> {
        (self.ttft_count > 0).then(|| self.ttft_ms_sum as f64 / self.ttft_count as f64)
    }

    /// Mean tokens per second across the hour. Derived from the sums, so a long
    /// slow request and a short fast one are weighted by their real cost rather
    /// than counted equally.
    pub fn avg_tok_per_sec(&self) -> Option<f64> {
        (self.total_ms_sum > 0 && self.completion_tokens > 0)
            .then(|| self.completion_tokens as f64 * 1000.0 / self.total_ms_sum as f64)
    }
}

/// Truncate a Unix-ms timestamp to the start of its hour.
pub fn hour_start(now_ms: u64) -> u64 {
    const HOUR_MS: u64 = 3_600_000;
    now_ms - (now_ms % HOUR_MS)
}

/// The rollup series, newest last.
#[derive(Default)]
pub struct PerfHistory {
    buckets: std::sync::Mutex<std::collections::VecDeque<HourBucket>>,
}

impl PerfHistory {
    /// Load a previously persisted series. A missing or unreadable key yields an
    /// empty history — losing a trend must never stop the daemon starting.
    pub fn load(db: &crate::storage::db::Database) -> Self {
        let buckets: std::collections::VecDeque<HourBucket> = db
            .get_json::<Vec<HourBucket>>("metrics", DB_KEY)
            .ok()
            .flatten()
            .map(|v| v.into_iter().collect())
            .unwrap_or_default();
        Self {
            buckets: std::sync::Mutex::new(buckets),
        }
    }

    /// Fold a completed request into the current hour's bucket.
    ///
    /// Returns `true` when this call started a NEW hour, which is the caller's
    /// cue to persist — writing on every request would put a redb transaction on
    /// the completion path.
    pub fn record(&self, snap: &crate::inference::trace::TraceSnapshot, now_ms: u64) -> bool {
        let hour = hour_start(now_ms);
        let mut g = match self.buckets.lock() {
            Ok(g) => g,
            Err(e) => e.into_inner(),
        };
        let rolled = match g.back() {
            Some(b) if b.hour_start_ms == hour => false,
            _ => {
                if g.len() >= MAX_HOURLY_BUCKETS {
                    g.pop_front();
                }
                g.push_back(HourBucket {
                    hour_start_ms: hour,
                    ..Default::default()
                });
                true
            }
        };
        // Safety: the arm above guarantees a back element.
        let b = g.back_mut().expect("bucket pushed above");
        b.requests += 1;
        b.total_ms_sum += snap.total_ms;
        b.completion_tokens += snap.completion_tokens as u64;
        if let Some(t) = snap.ttft_ms {
            b.ttft_ms_sum += t;
            b.ttft_count += 1;
        }
        if !matches!(snap.outcome, crate::inference::trace::Outcome::Ok) {
            b.errors += 1;
        }
        *b.by_route
            .entry(snap.route.as_str().to_string())
            .or_insert(0) += 1;
        rolled
    }

    /// Snapshot for rendering, oldest first.
    pub fn snapshot(&self) -> Vec<HourBucket> {
        match self.buckets.lock() {
            Ok(g) => g.iter().cloned().collect(),
            Err(e) => e.into_inner().iter().cloned().collect(),
        }
    }

    /// Persist the series. Best-effort: a write failure is logged, never
    /// propagated into the request path.
    pub fn persist(&self, db: &crate::storage::db::Database) {
        let v = self.snapshot();
        if let Err(e) = db.put_json("metrics", DB_KEY, &v) {
            tracing::debug!(error = %e, "could not persist hourly performance rollups");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::trace::{Outcome, RequestTrace, Route};

    fn snap(total_ms: u64, tokens: u32, ok: bool) -> crate::inference::trace::TraceSnapshot {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_assembled(Route::Distributed, vec![]);
        t.mark_first_token();
        t.mark_finished(
            if ok {
                Outcome::Ok
            } else {
                Outcome::Error("PipelineError".into())
            },
            1,
            tokens,
        );
        let mut s = t.snapshot();
        // Override the measured duration so the arithmetic under test is exact.
        s.total_ms = total_ms;
        s.ttft_ms = Some(100);
        s
    }

    const H: u64 = 3_600_000;

    #[test]
    fn same_hour_accumulates_into_one_bucket() {
        let h = PerfHistory::default();
        assert!(
            h.record(&snap(1000, 10, true), H),
            "first call opens a bucket"
        );
        assert!(!h.record(&snap(3000, 20, true), H + 60_000), "same hour");
        let b = h.snapshot();
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].requests, 2);
        assert_eq!(b[0].total_ms_sum, 4000);
        assert_eq!(b[0].completion_tokens, 30);
        assert_eq!(b[0].avg_total_ms(), Some(2000.0));
    }

    #[test]
    fn crossing_an_hour_opens_a_new_bucket_and_signals_persist() {
        let h = PerfHistory::default();
        h.record(&snap(1000, 1, true), H);
        assert!(
            h.record(&snap(1000, 1, true), 2 * H),
            "new hour must signal"
        );
        assert_eq!(h.snapshot().len(), 2);
    }

    #[test]
    fn errors_are_counted_separately_from_requests() {
        let h = PerfHistory::default();
        h.record(&snap(500, 5, true), H);
        h.record(&snap(500, 0, false), H);
        let b = &h.snapshot()[0];
        assert_eq!(b.requests, 2, "an error is still a request");
        assert_eq!(b.errors, 1);
    }

    #[test]
    fn throughput_weights_by_duration_not_by_request() {
        // 10 tokens in 1s and 10 tokens in 9s is 20 tokens in 10s = 2 tok/s,
        // NOT the mean of 10 and 1.1.
        let h = PerfHistory::default();
        h.record(&snap(1000, 10, true), H);
        h.record(&snap(9000, 10, true), H);
        let got = h.snapshot()[0].avg_tok_per_sec().unwrap();
        assert!((got - 2.0).abs() < 1e-9, "got {got}");
    }

    #[test]
    fn series_is_capped_at_a_week() {
        let h = PerfHistory::default();
        for i in 0..(MAX_HOURLY_BUCKETS as u64 + 10) {
            h.record(&snap(100, 1, true), i * H);
        }
        let b = h.snapshot();
        assert_eq!(b.len(), MAX_HOURLY_BUCKETS);
        // Oldest-first eviction: the surviving window is the most recent one.
        assert_eq!(b[0].hour_start_ms, 10 * H);
    }

    #[test]
    fn ttft_average_ignores_requests_that_reported_none() {
        let h = PerfHistory::default();
        h.record(&snap(1000, 5, true), H);
        let mut without = snap(1000, 5, true);
        without.ttft_ms = None;
        h.record(&without, H);
        let b = &h.snapshot()[0];
        assert_eq!(b.requests, 2);
        assert_eq!(b.ttft_count, 1, "only one request reported TTFT");
        assert_eq!(b.avg_ttft_ms(), Some(100.0), "must not divide by 2");
    }

    #[test]
    fn hour_start_truncates() {
        assert_eq!(hour_start(H + 12345), H);
        assert_eq!(hour_start(H), H);
    }
}
