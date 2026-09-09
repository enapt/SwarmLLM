//! Coordinator-side memory of what it has sent to each pipeline segment, so a
//! stand-in taking one over can be brought up to the failed machine's state
//! instead of inheriting nothing.
//!
//! **Why it exists.** A segment's KV cache is keyed by `(layer range, request
//! id)`, so a machine that has not served this segment for this request holds
//! nothing, and `failover_segment` re-sends only the CURRENT step. Measured
//! with `examples/failover_kv_probe.rs` on llama-3.2-3b: replacing 5 of 28
//! layers took the probability of the token the healthy machine would have
//! chosen from 0.997 to 0.119, and half the model took it to 0.005. Nothing
//! errors and nothing warns — the reply simply stops being the model's. So
//! `failover_can_restore_state` refuses mid-reply, and a reply already under
//! way ends rather than drifting.
//!
//! This is what turns that refusal back into a recovery. The coordinator drives
//! every hop, so it already SEES each segment's input; keeping those inputs
//! lets a replacement be replayed them as one prefill at position 0 — an
//! ordinary forward, needing no new message type — after which it holds the
//! same cache the failed machine had. Measured in the same probe: the replayed
//! stand-in reaches P = 0.9965 against the intact machine's 0.9966, at every
//! split tried (5, 14 and 21 of 28 layers).
//!
//! **Prior art.** Petals solves the identical failure with the identical shape:
//! servers keep "past attention keys and values for their layers", the client
//! keeps "past inputs sent to a given pipeline stage", and on a disconnect the
//! client "can find another server with that pipeline stage and use client-side
//! cache to restore the server state" — O(t) bytes in one round, recomputing
//! only the failed stages rather than re-running the pipeline (*Distributed
//! Inference and Fine-tuning of Large Language Models Over The Internet*,
//! arXiv 2312.08361, Algorithm 3). Retaining the boundary input is also the
//! cheaper of the two things one could keep: it is ONE hidden vector per
//! position, where the segment's KV is `2 × layers × kv_dim` — 12 KB against
//! 32 KB per position for a 4-layer segment of llama-3.2-3b, widening linearly
//! with segment size, and needing no traffic until something actually fails.
//!
//! **Where this deliberately differs from Petals**, whose client cache is
//! unbounded and persists "throughout inference": everything here is bounded,
//! because this node is a server for other people's traffic as well as a client
//! for its own.
//!
//! **The one property everything else serves: a partial history is never
//! replayed.** A replay assembled from a history with a hole in it rebuilds a
//! cache that is plausible and wrong, and — exactly like the defect this fixes
//! — nothing downstream could tell. So every way of losing a step marks the
//! segment unrestorable rather than shortening what is kept: exceeding the
//! budget, a chained run the coordinator never saw the middle of, a
//! tensor-parallel segment driven elsewhere, or a step whose position does not
//! continue the one before it. `restorable_history` returns `None` unless the
//! history is contiguous from position 0 to the position being restored, and
//! that is checked against the recorded spans rather than assumed.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use dashmap::DashMap;
use uuid::Uuid;

/// Requests retained at once. Past this, the least recently active is dropped —
/// it loses only the ABILITY to fail over mid-reply, never the request itself.
pub(crate) const MAX_RETAINED_REQUESTS: usize = 32;

/// Bytes of retained activation one request may hold across all its segments.
///
/// A position costs `hidden × 4` bytes per protected segment in the f32 wire
/// form (12 KB for llama-3.2-3b's 3072-wide hidden state, ~3.2 KB in the Q8_0
/// form), so this is roughly a 5k-token conversation with two protected
/// segments, or a 20k-token one with two if the sender quantises. Past it the
/// request stops being restorable rather than being restored from a hole.
pub(crate) const MAX_RETAINED_BYTES_PER_REQUEST: usize = 128 * 1024 * 1024;

/// Bytes retained across every request at once. The backstop that matters on a
/// small machine, where per-request budgets can still sum past what is there.
pub(crate) const MAX_RETAINED_BYTES_TOTAL: usize = 512 * 1024 * 1024;

/// How long a request's history outlives its last forward. Requests are
/// normally released explicitly by `release_request_state`; this only catches
/// one that ended in a way that skipped it.
pub(crate) const RETAINED_ACTIVATION_TTL: Duration = Duration::from_secs(300);

/// One segment's history for one request: the inputs sent to it, in order.
#[derive(Debug, Default)]
struct SegmentHistory {
    /// Wire-format activation buffers, one per forward, in send order. Kept
    /// encoded rather than decoded: it is what arrived, it is what will be
    /// sent, and for a Q8_0 sender it is ~3.8x smaller than the decoded form.
    steps: Vec<Vec<u8>>,
    /// Position the next step must start at for the history to stay
    /// contiguous. Compared rather than trusted — see the module doc.
    next_index_pos: u32,
    /// Once false, never true again for this request. A segment that has lost
    /// a step cannot be restored, and pretending otherwise is the defect this
    /// module exists to prevent.
    contiguous: bool,
    bytes: usize,
}

/// Everything retained for one request.
#[derive(Debug, Default)]
struct RequestActivations {
    segments: HashMap<usize, SegmentHistory>,
    bytes: usize,
    last_activity: Option<Instant>,
}

/// The store. One per node, on the root `SharedState`.
#[derive(Debug, Default)]
pub(crate) struct RetainedActivations {
    inner: DashMap<Uuid, RequestActivations>,
    /// Bytes across every request, maintained alongside the per-request totals
    /// so the global cap can be enforced without walking the map.
    total_bytes: std::sync::atomic::AtomicUsize,
}

impl RetainedActivations {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the input just sent to `segment_idx` at `index_pos`.
    ///
    /// `protect` is the caller's answer to "is this segment worth retaining for"
    /// — today, whether a standby covers it. A segment nothing can take over
    /// gains nothing from being restorable, and retaining it would spend the
    /// budget that protects the segments that can.
    ///
    /// Never fails and never blocks the forward: the worst outcome is that the
    /// segment stops being restorable, which is where it started.
    pub(crate) fn record(
        &self,
        request_id: Uuid,
        segment_idx: usize,
        index_pos: u32,
        activations: &[u8],
        protect: bool,
    ) {
        let Some(span) = crate::inference::tensor_util::activation_positions(activations) else {
            // A buffer whose header we cannot read is a buffer we cannot prove
            // the span of, so the history stops being provably contiguous.
            self.mark_unrestorable(request_id, segment_idx);
            return;
        };
        if !protect {
            return;
        }
        if self.total_bytes.load(std::sync::atomic::Ordering::Relaxed) + activations.len()
            > MAX_RETAINED_BYTES_TOTAL
        {
            self.mark_unrestorable(request_id, segment_idx);
            return;
        }
        self.evict_if_over_request_cap(request_id);

        let mut entry = self.inner.entry(request_id).or_default();
        entry.last_activity = Some(Instant::now());
        let over_budget = entry.bytes + activations.len() > MAX_RETAINED_BYTES_PER_REQUEST;
        let seg = entry.segments.entry(segment_idx).or_insert(SegmentHistory {
            contiguous: true,
            ..Default::default()
        });
        if !seg.contiguous {
            return;
        }
        if over_budget || index_pos != seg.next_index_pos {
            // Either we cannot afford the next step or this one does not
            // continue the last. Both mean the same thing downstream.
            seg.contiguous = false;
            let freed = seg.bytes;
            seg.steps = Vec::new();
            seg.bytes = 0;
            entry.bytes = entry.bytes.saturating_sub(freed);
            self.total_bytes
                .fetch_sub(freed, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        seg.steps.push(activations.to_vec());
        seg.bytes += activations.len();
        seg.next_index_pos = index_pos + span;
        entry.bytes += activations.len();
        self.total_bytes
            .fetch_add(activations.len(), std::sync::atomic::Ordering::Relaxed);
    }

    /// State this segment can no longer be restored for this request, whatever
    /// is already held for it. Used where a forward bypasses the coordinator —
    /// a chained run's middle, a tensor-parallel segment — so the history has a
    /// hole the recorder never sees.
    pub(crate) fn mark_unrestorable(&self, request_id: Uuid, segment_idx: usize) {
        let mut entry = self.inner.entry(request_id).or_default();
        entry.last_activity = Some(Instant::now());
        let seg = entry.segments.entry(segment_idx).or_default();
        let freed = seg.bytes;
        seg.contiguous = false;
        seg.steps = Vec::new();
        seg.bytes = 0;
        entry.bytes = entry.bytes.saturating_sub(freed);
        self.total_bytes
            .fetch_sub(freed, std::sync::atomic::Ordering::Relaxed);
    }

    /// The history to replay onto a stand-in for `segment_idx`, or `None` if
    /// there is not a provably complete one.
    ///
    /// `up_to_index_pos` is the position the takeover step sits at, and the
    /// history must cover exactly `0..up_to_index_pos` — checked, because "we
    /// kept something" and "we kept everything" are different claims and only
    /// the second one may be acted on.
    pub(crate) fn restorable_history(
        &self,
        request_id: Uuid,
        segment_idx: usize,
        up_to_index_pos: u32,
    ) -> Option<Vec<Vec<u8>>> {
        let entry = self.inner.get(&request_id)?;
        let seg = entry.segments.get(&segment_idx)?;
        if !seg.contiguous || seg.steps.is_empty() || seg.next_index_pos != up_to_index_pos {
            return None;
        }
        Some(seg.steps.clone())
    }

    /// Everything this request retained, dropped. Called by
    /// `release_request_state` beside the other per-request maps.
    pub(crate) fn release(&self, request_id: Uuid) {
        if let Some((_, entry)) = self.inner.remove(&request_id) {
            self.total_bytes
                .fetch_sub(entry.bytes, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Drop requests idle past the TTL. Wired to the health tick, for a request
    /// that ended without reaching `release_request_state`.
    pub(crate) fn sweep_stale(&self, ttl: Duration) {
        let now = Instant::now();
        let stale: Vec<Uuid> = self
            .inner
            .iter()
            .filter(|e| e.last_activity.is_none_or(|t| now.duration_since(t) > ttl))
            .map(|e| *e.key())
            .collect();
        for id in stale {
            self.release(id);
        }
    }

    /// Bytes held right now, for the diagnostics report and the tests.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.total_bytes.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Requests held right now.
    pub(crate) fn retained_requests(&self) -> usize {
        self.inner.len()
    }

    /// Make room for `request_id` if the map is already full of OTHER requests.
    fn evict_if_over_request_cap(&self, request_id: Uuid) {
        if self.inner.len() < MAX_RETAINED_REQUESTS || self.inner.contains_key(&request_id) {
            return;
        }
        let oldest = self
            .inner
            .iter()
            .min_by_key(|e| e.last_activity)
            .map(|e| *e.key());
        if let Some(id) = oldest {
            self.release(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A forward's wire bytes for `positions` positions of a 4-wide hidden
    /// state — the real encoder, so the header the recorder reads is the real
    /// header.
    fn forward_bytes(positions: usize) -> Vec<u8> {
        let t = candle_core::Tensor::zeros(
            (1, positions, 4),
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .unwrap();
        crate::inference::tensor_util::tensor_to_bytes(&t).unwrap()
    }

    #[test]
    fn a_contiguous_history_is_restorable_and_covers_every_position() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        // A 7-token prompt pass, then three decode steps.
        r.record(id, 1, 0, &forward_bytes(7), true);
        for pos in 7..10 {
            r.record(id, 1, pos, &forward_bytes(1), true);
        }
        let history = r
            .restorable_history(id, 1, 10)
            .expect("a history covering 0..10 is restorable");
        assert_eq!(history.len(), 4, "prompt pass plus three decode steps");
        // Asked for a position the history does not reach, it declines rather
        // than handing back what it has.
        assert!(r.restorable_history(id, 1, 11).is_none());
        assert!(r.restorable_history(id, 1, 9).is_none());
    }

    /// The property the module exists for: anything that loses a step makes the
    /// segment unrestorable, rather than making the replay shorter.
    #[test]
    fn a_history_with_a_hole_is_never_offered_for_replay() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        r.record(id, 0, 0, &forward_bytes(7), true);
        // A step that does not continue the last one — a forward the
        // coordinator did not see, a chained run, a re-send at the wrong place.
        r.record(id, 0, 9, &forward_bytes(1), true);
        assert!(
            r.restorable_history(id, 0, 10).is_none(),
            "a gap must make the segment unrestorable, not produce a short replay"
        );
        // And it stays that way: a later well-placed step does not heal it.
        r.record(id, 0, 10, &forward_bytes(1), true);
        assert!(r.restorable_history(id, 0, 11).is_none());
        assert_eq!(r.retained_bytes(), 0, "a hole releases what it was holding");
    }

    #[test]
    fn a_segment_nothing_can_take_over_is_not_retained() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        r.record(id, 0, 0, &forward_bytes(7), false);
        assert!(r.restorable_history(id, 0, 7).is_none());
        assert_eq!(r.retained_bytes(), 0);
    }

    #[test]
    fn a_bypassed_forward_marks_the_segment_unrestorable() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        r.record(id, 2, 0, &forward_bytes(7), true);
        assert!(r.restorable_history(id, 2, 7).is_some());
        // A chained run carried this segment; the coordinator never saw its
        // input, so what is held is no longer the whole story.
        r.mark_unrestorable(id, 2);
        assert!(r.restorable_history(id, 2, 7).is_none());
        assert_eq!(r.retained_bytes(), 0);
    }

    #[test]
    fn releasing_a_request_returns_every_byte_it_held() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        r.record(id, 0, 0, &forward_bytes(32), true);
        r.record(id, 1, 0, &forward_bytes(32), true);
        assert!(r.retained_bytes() > 0);
        assert_eq!(r.retained_requests(), 1);
        r.release(id);
        assert_eq!(r.retained_bytes(), 0, "the byte count follows the release");
        assert_eq!(r.retained_requests(), 0);
    }

    #[test]
    fn the_request_cap_evicts_the_least_recently_active() {
        let r = RetainedActivations::new();
        let ids: Vec<Uuid> = (0..MAX_RETAINED_REQUESTS + 4)
            .map(|_| Uuid::new_v4())
            .collect();
        for id in &ids {
            r.record(*id, 0, 0, &forward_bytes(1), true);
        }
        assert!(
            r.retained_requests() <= MAX_RETAINED_REQUESTS,
            "held {} requests against a cap of {MAX_RETAINED_REQUESTS}",
            r.retained_requests()
        );
        // The newest survived; the oldest did not.
        assert!(r.restorable_history(*ids.last().unwrap(), 0, 1).is_some());
        assert!(r.restorable_history(ids[0], 0, 1).is_none());
    }

    #[test]
    fn a_stale_request_is_swept() {
        let r = RetainedActivations::new();
        let id = Uuid::new_v4();
        r.record(id, 0, 0, &forward_bytes(4), true);
        r.sweep_stale(Duration::from_secs(3600));
        assert_eq!(r.retained_requests(), 1, "a fresh request is not swept");
        r.sweep_stale(Duration::from_millis(0));
        assert_eq!(r.retained_requests(), 0);
        assert_eq!(r.retained_bytes(), 0);
    }
}
