//! A requester's guesses for the splits this node leads (#318).
//!
//! A user holding none of a split model hands its request to the machine
//! holding the first layers (`remote_generate::try_delegated_split`), which
//! leads it here with what THIS node holds — usually no small model to guess
//! with. The requester often holds one. It follows the reply as it streams in
//! and sends windows of guesses (`types::RequesterGuesses`); the loop leading
//! the reply checks those that go past it each pass and never waits for one
//! (`pipeline::ngram_only_spec`). Both ends meet here:
//!
//! - the newest window, taken ONLY from the peer that handed the reply over,
//!   only while the reply runs ([`RequesterGuessSlots::offer_from`]);
//! - the reply's token ids, recorded by the loop as it accepts them
//!   ([`GuessSlot::record`]) and handed to the task streaming the reply
//!   ([`GuessSlot::take_unsent`]) — the stream carries text, and a drafter
//!   reads ids. The stream is handed its slot ([`OpenSlot::slot`]) and never
//!   looks one up by the requester's id: a requester picks that id, so a
//!   second hand-off could name one already open here, and a lookup would
//!   send it another reply's ids. That second hand-off gets no slot at all
//!   ([`RequesterGuessSlots::open`]).

use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use crate::types::{GuessesAsked, ReplyIds, RequesterGuesses, MAX_REQUESTER_GUESSES};

/// Does this node take a requester's guesses for the splits it leads? The one
/// answer, asked where a hand-off opens its slot
/// (`dispatch::remote_generate::handle_delegated_split`) AND where the node
/// advertises `features::REQUESTER_GUESSES` (`health::monitor`). Where its
/// owner turned speculation off it checks none — each costs this machine and
/// its peers rows — and a head advertising the bit anyway made every
/// requester load a drafter and read the whole prompt ahead for nothing.
pub(crate) fn takes_guesses(cfg: &crate::config::InferenceConfig) -> bool {
    cfg.speculative_decoding
}

/// One reply's guesses and ids.
pub(crate) struct GuessSlot {
    /// The peer that handed the reply over (its peer-id bytes) — the only one
    /// whose windows are taken.
    from: Vec<u8>,
    asked: GuessesAsked,
    inner: Mutex<SlotInner>,
}

#[derive(Default)]
struct SlotInner {
    window: Option<RequesterGuesses>,
    /// The reply's tokens as the loop accepted them.
    reply: Vec<u32>,
    /// How many of them have gone out with the stream.
    sent: usize,
}

/// What became of a window.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Offer {
    Taken,
    /// A newer window is already held: two messages crossed.
    Stale,
    Refused(&'static str),
}

impl GuessSlot {
    fn new(from: Vec<u8>, asked: GuessesAsked) -> Self {
        Self {
            from,
            asked,
            inner: Mutex::new(SlotInner::default()),
        }
    }

    /// What the requester said when it handed the reply over.
    pub(crate) fn asked(&self) -> &GuessesAsked {
        &self.asked
    }

    /// Up to `max` of the newest window's guesses that go on from `reply`
    /// (`RequesterGuesses::guesses_after`); none without a window.
    pub(crate) fn guesses(&self, reply: &[u32], max: usize) -> Vec<u32> {
        let Ok(inner) = self.inner.lock() else {
            return Vec::new();
        };
        inner
            .window
            .as_ref()
            .map(|w| w.guesses_after(reply, max).to_vec())
            .unwrap_or_default()
    }

    /// The reply's tokens `tokens` from position `from`, as the loop accepted
    /// them — BEFORE it emits them, so the stream finds them. A record that
    /// does not continue the one held (an attempt that started over) is
    /// ignored: the requester was sent the first one.
    pub(crate) fn record(&self, from: usize, tokens: &[u32]) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if from == inner.reply.len() {
            inner.reply.extend_from_slice(tokens);
        }
    }

    /// The ids recorded since the last call, for the next token streamed.
    pub(crate) fn take_unsent(&self) -> Option<ReplyIds> {
        let Ok(mut inner) = self.inner.lock() else {
            return None;
        };
        if inner.sent >= inner.reply.len() {
            return None;
        }
        let from = inner.sent;
        inner.sent = inner.reply.len();
        Some(ReplyIds {
            from: from as u32,
            ids: inner.reply[from..].to_vec(),
        })
    }

    fn offer(&self, window: RequesterGuesses) -> Offer {
        let Ok(mut inner) = self.inner.lock() else {
            return Offer::Refused("its slot is unusable");
        };
        // The requester drafts from the ids it was sent and from nothing else.
        if window.after as usize > inner.sent {
            return Offer::Refused("it follows more of the reply than was sent");
        }
        if inner
            .window
            .as_ref()
            .is_some_and(|held| held.after > window.after)
        {
            return Offer::Stale;
        }
        inner.window = Some(window);
        Offer::Taken
    }
}

/// Every reply this node leads for a requester that guesses, by the id the
/// requester knows it by (the hand-off's wire id) and by the id it runs under
/// here.
#[derive(Default)]
pub(crate) struct RequesterGuessSlots {
    by_wire: DashMap<uuid::Uuid, Arc<GuessSlot>>,
    by_led: DashMap<uuid::Uuid, Arc<GuessSlot>>,
}

/// An open slot; dropping it closes it, on every way the reply ends. It owns
/// both its ids while it lives: [`RequesterGuessSlots::open`] opens none over
/// an id already open.
pub(crate) struct OpenSlot {
    slots: Arc<RequesterGuessSlots>,
    slot: Arc<GuessSlot>,
    wire: uuid::Uuid,
    led: uuid::Uuid,
}

impl OpenSlot {
    /// The slot, for the task streaming the reply.
    pub(crate) fn slot(&self) -> &Arc<GuessSlot> {
        &self.slot
    }
}

impl Drop for OpenSlot {
    fn drop(&mut self) {
        self.slots.by_wire.remove(&self.wire);
        self.slots.by_led.remove(&self.led);
    }
}

impl RequesterGuessSlots {
    /// Open the slot of a reply the peer `from` handed over as `wire`, led
    /// here as `led`. The ONLY writer of the maps
    /// (`dispatch::remote_generate::handle_delegated_split`). `None` where
    /// either id is already open: the reply then goes on without guesses, and
    /// the open one keeps its own.
    pub(crate) fn open(
        self: &Arc<Self>,
        wire: uuid::Uuid,
        led: uuid::Uuid,
        from: Vec<u8>,
        asked: GuessesAsked,
    ) -> Option<OpenSlot> {
        use dashmap::mapref::entry::Entry;
        let slot = Arc::new(GuessSlot::new(from, asked));
        match self.by_wire.entry(wire) {
            Entry::Occupied(_) => return None,
            Entry::Vacant(v) => {
                v.insert(slot.clone());
            }
        }
        let led_taken = match self.by_led.entry(led) {
            Entry::Occupied(_) => true,
            Entry::Vacant(v) => {
                v.insert(slot.clone());
                false
            }
        };
        if led_taken {
            self.by_wire.remove(&wire);
            return None;
        }
        Some(OpenSlot {
            slots: self.clone(),
            slot,
            wire,
            led,
        })
    }

    /// A window that arrived from `sender` (its AUTHENTICATED peer-id bytes) —
    /// dispatch's `RequesterGuesses` arm, the only caller.
    pub(crate) fn offer_from(&self, sender: &[u8], window: RequesterGuesses) -> Offer {
        if window.tokens.len() > MAX_REQUESTER_GUESSES {
            return Offer::Refused("it carries more guesses than a window may");
        }
        let Some(slot) = self
            .by_wire
            .get(&window.request_id)
            .map(|s| s.value().clone())
        else {
            return Offer::Refused("no reply of that id is led here");
        };
        if slot.from != sender {
            return Offer::Refused("it is not from the peer the reply is for");
        }
        slot.offer(window)
    }

    /// The slot of the reply running here as `led` — the loop leading it.
    /// `led` is this node's own fresh id for the reply, never the requester's.
    pub(crate) fn for_led(&self, led: &uuid::Uuid) -> Option<Arc<GuessSlot>> {
        self.by_led.get(led).map(|s| s.value().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asked() -> GuessesAsked {
        GuessesAsked {
            coupling_seed: None,
            prompt_tokens: 10,
        }
    }

    fn window(id: uuid::Uuid, after: u32, tokens: &[u32]) -> RequesterGuesses {
        RequesterGuesses {
            request_id: id,
            after,
            tokens: tokens.to_vec(),
        }
    }

    /// A slot with the reply's first `sent` ids already streamed.
    fn open_with_reply(
        sent: &[u32],
    ) -> (Arc<RequesterGuessSlots>, OpenSlot, uuid::Uuid, uuid::Uuid) {
        let slots = Arc::new(RequesterGuessSlots::default());
        let (wire, led) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let open = slots.open(wire, led, vec![1, 2, 3], asked()).unwrap();
        let slot = slots.for_led(&led).unwrap();
        slot.record(0, sent);
        assert_eq!(slot.take_unsent().map(|r| r.ids), Some(sent.to_vec()));
        (slots, open, wire, led)
    }

    /// Guesses are taken from the peer the reply is for and from nobody else:
    /// another peer naming the reply's id is refused, its window never held.
    #[test]
    fn a_window_is_taken_only_from_the_peer_the_reply_is_for() {
        let (slots, _open, wire, led) = open_with_reply(&[5, 6]);
        assert_eq!(
            slots.offer_from(&[9, 9, 9], window(wire, 2, &[7, 8])),
            Offer::Refused("it is not from the peer the reply is for")
        );
        assert!(slots.for_led(&led).unwrap().guesses(&[5, 6], 8).is_empty());
        assert_eq!(
            slots.offer_from(&[1, 2, 3], window(wire, 2, &[7, 8])),
            Offer::Taken
        );
        assert_eq!(slots.for_led(&led).unwrap().guesses(&[5, 6], 8), vec![7, 8]);
    }

    /// A window over the cap is dropped unread; one following more of the
    /// reply than was sent — which no honest requester can draft — is refused.
    #[test]
    fn an_oversized_or_impossible_window_is_refused() {
        let (slots, _open, wire, _) = open_with_reply(&[5, 6]);
        let big = vec![1; MAX_REQUESTER_GUESSES + 1];
        assert!(matches!(
            slots.offer_from(&[1, 2, 3], window(wire, 2, &big)),
            Offer::Refused(_)
        ));
        assert!(matches!(
            slots.offer_from(&[1, 2, 3], window(wire, 3, &[1])),
            Offer::Refused(_)
        ));
    }

    /// Two windows that crossed on the way: the one drafted from less of the
    /// reply arrives second and must not replace the newer one.
    #[test]
    fn a_window_that_crossed_a_newer_one_does_not_replace_it() {
        let (slots, _open, wire, led) = open_with_reply(&[5, 6, 7]);
        assert_eq!(
            slots.offer_from(&[1, 2, 3], window(wire, 3, &[8, 9])),
            Offer::Taken
        );
        assert_eq!(
            slots.offer_from(&[1, 2, 3], window(wire, 2, &[7, 1])),
            Offer::Stale
        );
        assert_eq!(
            slots.for_led(&led).unwrap().guesses(&[5, 6, 7], 8),
            vec![8, 9]
        );
    }

    /// The stream is handed each id once, in order; an attempt that started
    /// over records nothing the requester would read as more of this reply.
    #[test]
    fn the_reply_ids_go_out_once_in_order() {
        let slots = Arc::new(RequesterGuessSlots::default());
        let (wire, led) = (uuid::Uuid::new_v4(), uuid::Uuid::new_v4());
        let open = slots.open(wire, led, vec![1], asked()).unwrap();
        let slot = open.slot();
        assert!(slot.take_unsent().is_none());
        slot.record(0, &[4]);
        slot.record(1, &[5, 6]);
        slot.record(0, &[9, 9]);
        assert_eq!(
            slot.take_unsent(),
            Some(ReplyIds {
                from: 0,
                ids: vec![4, 5, 6]
            })
        );
        assert!(slot.take_unsent().is_none());
        slot.record(3, &[7]);
        assert_eq!(
            slot.take_unsent(),
            Some(ReplyIds {
                from: 3,
                ids: vec![7]
            })
        );
    }

    /// The slot closes with its reply: a window for it afterwards finds nothing.
    #[test]
    fn a_slot_closes_when_its_reply_ends() {
        let (slots, open, wire, led) = open_with_reply(&[5]);
        drop(open);
        assert!(slots.for_led(&led).is_none());
        assert!(matches!(
            slots.offer_from(&[1, 2, 3], window(wire, 1, &[2])),
            Offer::Refused(_)
        ));
    }

    /// A requester picks the id it hands a reply over under, so a second
    /// hand-off may name one already open here. It gets no slot: the open
    /// reply keeps its guesses and its ids, and closing nothing of the second
    /// takes them away.
    #[test]
    fn a_second_reply_under_an_open_id_gets_no_slot() {
        let (slots, open, wire, led) = open_with_reply(&[5, 6]);
        assert!(slots
            .open(wire, uuid::Uuid::new_v4(), vec![9], asked())
            .is_none());
        assert!(slots
            .open(uuid::Uuid::new_v4(), led, vec![9], asked())
            .is_none());
        assert_eq!(
            slots.offer_from(&[1, 2, 3], window(wire, 2, &[7])),
            Offer::Taken,
            "the open reply still takes its own requester's windows"
        );
        assert!(Arc::ptr_eq(open.slot(), &slots.for_led(&led).unwrap()));
        let other = uuid::Uuid::new_v4();
        assert!(
            slots
                .open(other, uuid::Uuid::new_v4(), vec![9], asked())
                .is_some(),
            "a refused open leaves no id of its own behind"
        );
    }
}
