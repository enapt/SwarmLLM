//! Slot table for the worker's batched-generate decode pool (Item 7).
//!
//! Each `Slot` is one in-flight `Generate` request being driven through the
//! worker's slot-driven decode loop. Phase 1 admits a slot only after running
//! its full prefill in the admit handler — fast for one slot at a time, but a
//! long admission stalls every active decode slot for the duration of the
//! prefill.
//!
//! Phase 2 (Sarathi-style chunked prefill) replaces that with a state machine:
//! every slot starts in `Prefilling` (the prompt is held in the slot, the
//! admit handler does no compute). Each decode tick processes one prefill
//! chunk per `Prefilling` slot before running the batched decode over
//! `Decoding` slots — so a long admission can no longer block decode for
//! more than `prefill_chunk_tokens` of compute.
//!
//! The slot table itself is single-threaded — it lives in `run_worker`'s
//! task and is mutated under direct ownership.

use uuid::Uuid;

use crate::types::SamplingParams;

/// Per-slot state machine.
///
/// `Prefilling` slots haven't sampled their first token yet — every decode
/// tick advances them by `prefill_chunk_tokens` until the prompt is consumed,
/// then samples and transitions to `Decoding` in the same tick.
pub enum SlotState {
    /// Prompt is being prefilled in chunks.
    ///
    /// `remaining_ids` are the prompt tokens still to feed into the model
    /// (already advanced past any prefix-cache hit). `next_chunk_index_pos`
    /// is the KV-cache position the next chunk will write into (== prompt
    /// position of the first token in `remaining_ids`).
    Prefilling {
        remaining_ids: Vec<u32>,
        next_chunk_index_pos: usize,
    },
    /// Prompt prefill is complete; slot is in the per-token decode loop.
    Decoding {
        /// Next token to feed into the model on the upcoming decode tick.
        last_token: u32,
        /// Optional logprob of `last_token` collected at sample time.
        last_token_logprob: Option<f32>,
        /// Number of `Token` IPC messages emitted to the daemon so far.
        generated_count: usize,
        /// KV-cache write position for the upcoming decode forward.
        index_pos: usize,
    },
}

/// One active in-flight `Generate` inside the worker.
pub struct Slot {
    pub request_id: Uuid,
    pub req_id_str: String,
    /// KV-cache key — same string `SplitModel::kv_model_key()` returns. Used
    /// by `KvCacheStore` lookups inside `forward_batch`.
    pub model_key: String,
    /// (layer_start, layer_end) — every slot in a single SlotTable shares
    /// these so they all dispatch to the same `models[(start, end, 0, 1)]`
    /// variant.
    pub layer_range: (usize, usize),
    pub state: SlotState,
    pub max_tokens: u32,
    pub use_logprobs: bool,
    pub eos: Vec<u32>,
    pub stop_sequences: Vec<String>,
    pub accumulated_text: String,
    pub sampling: SamplingParams,
    /// Total prompt token count — set at admission, used in `GenerateDone`.
    pub prompt_tokens: usize,
    /// Full prompt token IDs — held so the prefix-cache can snapshot the
    /// completed prefill back into the radix tree for future hits. Cleared
    /// once the snapshot is inserted.
    pub prompt_ids: Vec<u32>,
    /// Set when the slot decides to stop. The slot driver reads this on
    /// the next pass and removes the slot. Possible values:
    /// - `"stop"`  — EOS or stop-sequence match
    /// - `"length"` — generated_count reached max_tokens
    /// - `"error"` — per-slot forward / sample failure (see `error_message`)
    pub finish_reason: Option<&'static str>,
    /// When `finish_reason == Some("error")`, the human-readable message that
    /// gets emitted to the daemon as `WorkerMsg::Error`. Lets one bad slot
    /// fail in isolation without aborting its neighbors.
    pub error_message: Option<String>,
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

    /// Mark `error` finish with a message. First-write-wins like the other
    /// finish helpers, so a per-slot decode error doesn't get clobbered by
    /// a downstream length check on the same tick.
    pub fn finish_error(&mut self, message: impl Into<String>) {
        if self.finish_reason.is_none() {
            self.finish_reason = Some("error");
            self.error_message = Some(message.into());
        }
    }

    pub fn is_prefilling(&self) -> bool {
        matches!(self.state, SlotState::Prefilling { .. })
    }

    pub fn is_decoding(&self) -> bool {
        matches!(self.state, SlotState::Decoding { .. })
    }

    /// `generated_count` is only meaningful in `Decoding` state — `Prefilling`
    /// hasn't emitted any tokens yet.
    pub fn generated_count(&self) -> usize {
        match &self.state {
            SlotState::Decoding {
                generated_count, ..
            } => *generated_count,
            SlotState::Prefilling { .. } => 0,
        }
    }

    /// Pop up to `chunk_size` tokens from the front of `remaining_ids`. Caller
    /// must verify the slot is `Prefilling` first.
    ///
    /// Returns `(chunk, chunk_index_pos, remaining_after_chunk)` so the caller
    /// can run one forward over `chunk` at `chunk_index_pos` and immediately
    /// know whether this was the final chunk (`remaining_after_chunk == 0`).
    pub fn take_prefill_chunk(&mut self, chunk_size: usize) -> Option<(Vec<u32>, usize, usize)> {
        let chunk_size = chunk_size.max(1);
        match &mut self.state {
            SlotState::Prefilling {
                remaining_ids,
                next_chunk_index_pos,
            } => {
                if remaining_ids.is_empty() {
                    return None;
                }
                let take = chunk_size.min(remaining_ids.len());
                let chunk: Vec<u32> = remaining_ids.drain(..take).collect();
                let pos = *next_chunk_index_pos;
                *next_chunk_index_pos += take;
                let remaining_after = remaining_ids.len();
                Some((chunk, pos, remaining_after))
            }
            SlotState::Decoding { .. } => None,
        }
    }

    /// Transition `Prefilling` → `Decoding` after the final chunk has been
    /// processed and the first decode token has been sampled. The new
    /// `index_pos` is `prompt_tokens` (KV cache now holds every prompt
    /// position).
    pub fn promote_to_decoding(&mut self, first_token: u32, first_logprob: Option<f32>) {
        let new_index_pos = match &self.state {
            SlotState::Prefilling {
                next_chunk_index_pos,
                ..
            } => *next_chunk_index_pos,
            SlotState::Decoding { index_pos, .. } => *index_pos,
        };
        self.state = SlotState::Decoding {
            last_token: first_token,
            last_token_logprob: first_logprob,
            generated_count: 0,
            index_pos: new_index_pos,
        };
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

    fn dummy_decoding_slot(rid: Uuid, layer_range: (usize, usize)) -> Slot {
        Slot {
            request_id: rid,
            req_id_str: rid.to_string(),
            model_key: "test-key".to_string(),
            layer_range,
            state: SlotState::Decoding {
                last_token: 42,
                last_token_logprob: None,
                generated_count: 0,
                index_pos: 5,
            },
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
            prompt_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            finish_reason: None,
            error_message: None,
        }
    }

    fn dummy_prefilling_slot(
        rid: Uuid,
        layer_range: (usize, usize),
        remaining_ids: Vec<u32>,
        prompt_len: usize,
    ) -> Slot {
        let prefilled = prompt_len.saturating_sub(remaining_ids.len());
        Slot {
            request_id: rid,
            req_id_str: rid.to_string(),
            model_key: "test-key".to_string(),
            layer_range,
            state: SlotState::Prefilling {
                remaining_ids,
                next_chunk_index_pos: prefilled,
            },
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
            prompt_tokens: prompt_len,
            prompt_ids: (1..=prompt_len as u32).collect(),
            finish_reason: None,
            error_message: None,
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
        table.admit(dummy_decoding_slot(Uuid::new_v4(), (0, 22)));
        assert_eq!(table.current_layer_range(), Some((0, 22)));
        assert!(table.can_admit((0, 22)));
        assert!(!table.can_admit((0, 32)));
        assert!(!table.can_admit((10, 22)));
    }

    #[test]
    fn full_table_rejects_admission() {
        let mut table = SlotTable::new(2);
        table.admit(dummy_decoding_slot(Uuid::new_v4(), (0, 22)));
        table.admit(dummy_decoding_slot(Uuid::new_v4(), (0, 22)));
        assert!(table.is_full());
        assert!(!table.can_admit((0, 22)));
    }

    #[test]
    fn drain_finished_returns_finished_and_keeps_active() {
        let mut table = SlotTable::new(4);
        let r1 = Uuid::new_v4();
        let r2 = Uuid::new_v4();
        let r3 = Uuid::new_v4();
        table.admit(dummy_decoding_slot(r1, (0, 22)));
        table.admit(dummy_decoding_slot(r2, (0, 22)));
        table.admit(dummy_decoding_slot(r3, (0, 22)));
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
        table.admit(dummy_decoding_slot(Uuid::new_v4(), (0, 22)));
        table.active()[0].finish_stop();
        let _ = table.drain_finished();
        assert!(table.is_empty());
        assert_eq!(table.current_layer_range(), None);
        assert!(table.can_admit((10, 32)));
    }

    #[test]
    fn finish_length_only_sets_when_unset() {
        let mut s = dummy_decoding_slot(Uuid::new_v4(), (0, 22));
        s.finish_stop();
        s.finish_length();
        assert_eq!(s.finish_reason, Some("stop"));
    }

    #[test]
    fn prefill_chunk_advances_index_pos_and_drains_remaining() {
        let mut s = dummy_prefilling_slot(Uuid::new_v4(), (0, 22), vec![1, 2, 3, 4, 5], 5);
        let (chunk, pos, remaining_after) = s.take_prefill_chunk(2).unwrap();
        assert_eq!(chunk, vec![1, 2]);
        assert_eq!(pos, 0);
        assert_eq!(remaining_after, 3);
        let (chunk, pos, remaining_after) = s.take_prefill_chunk(2).unwrap();
        assert_eq!(chunk, vec![3, 4]);
        assert_eq!(pos, 2);
        assert_eq!(remaining_after, 1);
        let (chunk, pos, remaining_after) = s.take_prefill_chunk(2).unwrap();
        assert_eq!(chunk, vec![5]);
        assert_eq!(pos, 4);
        assert_eq!(remaining_after, 0);
        // Drained — next call returns None.
        assert!(s.take_prefill_chunk(2).is_none());
    }

    #[test]
    fn prefill_chunk_with_prefix_cache_hit_starts_at_prefix_len() {
        // Prompt is 8 tokens; prefix-cache matched 5 → remaining is the last 3,
        // and chunk_index_pos starts at the prefix length.
        let mut s = dummy_prefilling_slot(Uuid::new_v4(), (0, 22), vec![6, 7, 8], 8);
        let (chunk, pos, remaining_after) = s.take_prefill_chunk(8).unwrap();
        assert_eq!(chunk, vec![6, 7, 8]);
        assert_eq!(pos, 5);
        assert_eq!(remaining_after, 0);
    }

    #[test]
    fn prefill_chunk_caps_at_remaining_when_chunk_size_too_big() {
        let mut s = dummy_prefilling_slot(Uuid::new_v4(), (0, 22), vec![1, 2, 3], 3);
        let (chunk, pos, remaining_after) = s.take_prefill_chunk(1024).unwrap();
        assert_eq!(chunk, vec![1, 2, 3]);
        assert_eq!(pos, 0);
        assert_eq!(remaining_after, 0);
    }

    #[test]
    fn prefill_chunk_zero_size_treated_as_one() {
        let mut s = dummy_prefilling_slot(Uuid::new_v4(), (0, 22), vec![1, 2, 3], 3);
        let (chunk, _, remaining_after) = s.take_prefill_chunk(0).unwrap();
        assert_eq!(chunk, vec![1]);
        assert_eq!(remaining_after, 2);
    }

    #[test]
    fn promote_to_decoding_clears_remaining_and_seeds_first_token() {
        let mut s = dummy_prefilling_slot(Uuid::new_v4(), (0, 22), vec![], 8);
        // Simulate next_chunk_index_pos already at prompt_tokens (last chunk drained).
        if let SlotState::Prefilling {
            next_chunk_index_pos,
            ..
        } = &mut s.state
        {
            *next_chunk_index_pos = 8;
        }
        s.promote_to_decoding(99, Some(-2.5));
        assert!(s.is_decoding());
        match s.state {
            SlotState::Decoding {
                last_token,
                last_token_logprob,
                generated_count,
                index_pos,
            } => {
                assert_eq!(last_token, 99);
                assert_eq!(last_token_logprob, Some(-2.5));
                assert_eq!(generated_count, 0);
                assert_eq!(index_pos, 8);
            }
            _ => panic!("expected Decoding"),
        }
    }

    #[test]
    fn take_prefill_chunk_on_decoding_slot_returns_none() {
        let mut s = dummy_decoding_slot(Uuid::new_v4(), (0, 22));
        assert!(s.take_prefill_chunk(4).is_none());
    }

    #[test]
    fn finish_error_records_message_and_blocks_other_finishers() {
        let mut s = dummy_decoding_slot(Uuid::new_v4(), (0, 22));
        s.finish_error("forward failed: OOM");
        assert_eq!(s.finish_reason, Some("error"));
        assert_eq!(s.error_message.as_deref(), Some("forward failed: OOM"));
        // First-write-wins — a downstream length check shouldn't overwrite the error.
        s.finish_length();
        s.finish_stop();
        assert_eq!(s.finish_reason, Some("error"));
        assert_eq!(s.error_message.as_deref(), Some("forward failed: OOM"));
    }
}
