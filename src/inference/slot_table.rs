//! Slot table for the worker's batched-generate decode pool (Item 7).
//!
//! Each `Slot` is one in-flight `Generate` request that has finished its
//! prefill and is now in the decode loop. The worker batches all active
//! slots into a single `SplitModel::forward_batch` call per tick, sampling
//! per-slot from the returned logits.
//!
//! The slot table itself is single-threaded — it lives in `run_worker`'s
//! task and is mutated under direct ownership.

use uuid::Uuid;

use crate::types::SamplingParams;

/// One active decoding stream inside the worker.
///
/// `last_token` is the next token to feed to the model on the upcoming
/// decode step. `index_pos` is the position that token will write into
/// the per-request KV cache. After every step we push the just-emitted
/// token into `generated`, advance `index_pos`, and check EOS / stop /
/// max_tokens.
pub struct Slot {
    pub request_id: Uuid,
    pub req_id_str: String,
    /// KV-cache key — same string `SplitModel::kv_model_key()` returns. Used
    /// by `KvCacheStore` lookups inside `forward_batch`.
    pub model_key: String,
    /// (layer_start, layer_end) — every slot in a single SlotTable shares
    /// these so they all dispatch to the same `models[(start, end, 0, 1)]`
    /// variant when batched.
    pub layer_range: (usize, usize),
    pub index_pos: usize,
    pub last_token: u32,
    pub last_token_logprob: Option<f32>,
    pub generated_count: usize,
    pub max_tokens: u32,
    pub use_logprobs: bool,
    pub eos: Vec<u32>,
    pub stop_sequences: Vec<String>,
    pub accumulated_text: String,
    pub sampling: SamplingParams,
    pub prompt_tokens: usize,
    /// Set when the slot decides to stop. The slot driver reads this on
    /// the next pass and removes the slot.
    pub finish_reason: Option<&'static str>,
}

impl Slot {
    /// Has this slot decided to stop?
    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    /// Mark `length` finish — runs when generated_count reaches max_tokens.
    pub fn finish_length(&mut self) {
        if self.finish_reason.is_none() {
            self.finish_reason = Some("length");
        }
    }

    /// Mark `stop` finish — EOS or stop-sequence match.
    pub fn finish_stop(&mut self) {
        if self.finish_reason.is_none() {
            self.finish_reason = Some("stop");
        }
    }
}

/// Slot table for one worker subprocess.
///
/// All slots in a single table share `layer_range` (set by the first admit;
/// later admits with a different range are rejected — caller falls through
/// to the sequential `handle_generate` path).
pub struct SlotTable {
    slots: Vec<Slot>,
    layer_range: Option<(usize, usize)>,
    capacity: usize,
}

impl SlotTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity.max(1)),
            layer_range: None,
            capacity: capacity.max(1),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.slots.len() >= self.capacity
    }

    pub fn current_layer_range(&self) -> Option<(usize, usize)> {
        self.layer_range
    }

    /// Can a new slot with this `layer_range` join the table right now?
    /// Rejects when full or when the table is already pinned to a different
    /// layer range. Caller falls through to `handle_generate`.
    pub fn can_admit(&self, layer_range: (usize, usize)) -> bool {
        if self.is_full() {
            return false;
        }
        match self.layer_range {
            None => true,
            Some(existing) => existing == layer_range,
        }
    }

    /// Insert a slot. Caller must have checked `can_admit`.
    pub fn admit(&mut self, slot: Slot) {
        let lr = slot.layer_range;
        self.slots.push(slot);
        if self.layer_range.is_none() {
            self.layer_range = Some(lr);
        }
    }

    /// Mutable slice of currently-active slots — drives the decode tick.
    pub fn active(&mut self) -> &mut [Slot] {
        &mut self.slots
    }

    /// Take ownership of every active slot, leaving the table empty. Used by
    /// the worker on a fatal decode error so each slot can be reported to its
    /// caller individually.
    pub fn into_active(self) -> Vec<Slot> {
        self.slots
    }

    /// Drain finished slots, leaving the table populated with the active
    /// remainder. When the table empties, the pinned layer range clears too
    /// so the next admission can pin a different range.
    pub fn drain_finished(&mut self) -> Vec<Slot> {
        let mut finished: Vec<Slot> = Vec::new();
        let mut i = 0;
        while i < self.slots.len() {
            if self.slots[i].is_finished() {
                finished.push(self.slots.swap_remove(i));
            } else {
                i += 1;
            }
        }
        if self.slots.is_empty() {
            self.layer_range = None;
        }
        finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SamplingParams;

    fn dummy_slot(rid: Uuid, layer_range: (usize, usize)) -> Slot {
        Slot {
            request_id: rid,
            req_id_str: rid.to_string(),
            model_key: "test-key".to_string(),
            layer_range,
            index_pos: 5,
            last_token: 42,
            last_token_logprob: None,
            generated_count: 0,
            max_tokens: 16,
            use_logprobs: false,
            eos: vec![2],
            stop_sequences: vec![],
            accumulated_text: String::new(),
            sampling: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                max_tokens: 16,
                stop: vec![],
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logprobs: false,
                top_logprobs: 0,
            },
            prompt_tokens: 8,
            finish_reason: None,
        }
    }

    #[test]
    fn empty_table_admits_any_layer_range() {
        let table = SlotTable::new(4);
        assert!(table.is_empty());
        assert!(table.can_admit((0, 22)));
        assert!(table.can_admit((10, 32)));
        assert_eq!(table.current_layer_range(), None);
    }

    #[test]
    fn first_admit_pins_layer_range_and_blocks_others() {
        let mut table = SlotTable::new(4);
        table.admit(dummy_slot(Uuid::new_v4(), (0, 22)));
        assert_eq!(table.current_layer_range(), Some((0, 22)));
        assert!(table.can_admit((0, 22)));
        assert!(!table.can_admit((0, 32)));
        assert!(!table.can_admit((10, 22)));
    }

    #[test]
    fn full_table_rejects_admission() {
        let mut table = SlotTable::new(2);
        table.admit(dummy_slot(Uuid::new_v4(), (0, 22)));
        table.admit(dummy_slot(Uuid::new_v4(), (0, 22)));
        assert!(table.is_full());
        assert!(!table.can_admit((0, 22)));
    }

    #[test]
    fn drain_finished_returns_finished_and_keeps_active() {
        let mut table = SlotTable::new(4);
        let r1 = Uuid::new_v4();
        let r2 = Uuid::new_v4();
        let r3 = Uuid::new_v4();
        table.admit(dummy_slot(r1, (0, 22)));
        table.admit(dummy_slot(r2, (0, 22)));
        table.admit(dummy_slot(r3, (0, 22)));
        table.active()[1].finish_stop();
        let finished = table.drain_finished();
        assert_eq!(finished.len(), 1);
        assert_eq!(finished[0].request_id, r2);
        assert_eq!(table.len(), 2);
        let remaining: Vec<Uuid> = table.active().iter().map(|s| s.request_id).collect();
        assert!(remaining.contains(&r1));
        assert!(remaining.contains(&r3));
    }

    #[test]
    fn drain_to_empty_releases_layer_range_pin() {
        let mut table = SlotTable::new(4);
        table.admit(dummy_slot(Uuid::new_v4(), (0, 22)));
        table.active()[0].finish_stop();
        let _ = table.drain_finished();
        assert!(table.is_empty());
        assert_eq!(table.current_layer_range(), None);
        assert!(table.can_admit((10, 32)));
    }

    #[test]
    fn finish_length_only_sets_when_unset() {
        let mut s = dummy_slot(Uuid::new_v4(), (0, 22));
        s.finish_stop();
        s.finish_length();
        assert_eq!(s.finish_reason, Some("stop"));
    }
}
