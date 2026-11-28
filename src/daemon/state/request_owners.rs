//! Who may end a conversation this node serves for another computer (#512).
//!
//! A far computer keeps a split conversation's cache keyed by the request id
//! it was sent with. The id is a random UUID, so a stranger cannot name one —
//! but every computer of the same chain sees it on the wire, and before this a
//! `CancelInference` from ANY authenticated computer naming it aborted the
//! conversation's work here and dropped its cache. A peer inside the chain
//! could end another computer's reply, step after step.
//!
//! The one sender a cancel legitimately comes from is the computer that
//! coordinates the conversation (`router::release_request_on_peers`, sent
//! straight from it). So this node remembers, for each conversation it serves,
//! the coordinator its FIRST admitted forward names
//! (`layer_forward::judged_as`: the chained forward's named coordinator, else
//! the sender), and honours a cancel only from that computer. The first name
//! stands: a later forward naming another coordinator does not move it — the
//! rule a chain hop's own report already keeps (`hop_timings::HopHeld`, #427).
//!
//! **No peer can push another's records out** (security review of this
//! change, 2026-11-28): with one shared bound, a peer opening ~4,096 ids of its
//! own evicted a live conversation's record, then named itself its owner. So
//! each SENDER of forwards holds at most [`MOST_A_SENDER`] records (the share
//! `HopHeld` gives a peer, `MOST_REQUESTS_A_PEER`), and a full table or share
//! gives up its least recently SEEN record — a live conversation's forwards
//! keep its own fresh.
//!
//! **Not closed here**: a chain hop that names ITSELF as coordinator on the
//! first forward a computer after it sees is recorded as the owner there. It
//! has already broken that reply (the computers after it answer it, not the
//! coordinator); what it keeps is the cancel for that conversation there until
//! the record ages out. Closing it needs the coordinator to sign
//! `(request_id, chain)` (FUTURE_WORK #512).
//!
//! Prior art: WireGuard binds every session to the peer's static key, and
//! QUIC's connection ids belong to one connection — an id another endpoint
//! learns is never a handle it can act through.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::types::NodeId;

/// Conversations remembered at once.
pub(crate) const MOST_REMEMBERED: usize = 4096;

/// Conversations one sender of forwards may have remembered at once — as
/// `hop_timings::MOST_REQUESTS_A_PEER`.
pub(crate) const MOST_A_SENDER: usize = 64;

/// How long a conversation is remembered after its last forward here. A
/// coordinator's cancel goes out when its reply ends, seconds after its last
/// forward (a hand-off is noted again when its reply ends); a reply quiet this
/// long has already been released by the worker's own idle eviction.
pub(crate) const REMEMBERED_FOR: Duration = Duration::from_secs(30 * 60);

#[derive(Default)]
pub(crate) struct RequestOwners {
    by_request: parking_lot::Mutex<HashMap<uuid::Uuid, Record>>,
}

struct Record {
    owner: NodeId,
    /// The computer whose forward made the record: its share is charged.
    sent_by: NodeId,
    seen: Instant,
}

/// What a cancel's sender is to the conversation it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CancelFrom {
    /// The computer that coordinates it: honour the cancel.
    Owner,
    /// Another computer: ignore the cancel.
    NotTheOwner,
    /// No forward of it was admitted here (or too long ago): nothing of it is
    /// held to cancel on this computer's say-so.
    Unknown,
}

impl RequestOwners {
    /// Record the coordinator a forward of `request_id` from `sender` names,
    /// at `now`. The first name stands; a forward naming the same one keeps it
    /// fresh.
    pub(crate) fn note_forward(
        &self,
        request_id: uuid::Uuid,
        coordinator: NodeId,
        sender: &NodeId,
        now: Instant,
    ) {
        let mut by_request = self.by_request.lock();
        if let Some(record) = by_request.get_mut(&request_id) {
            if record.owner == coordinator {
                record.seen = now;
            }
            return;
        }
        let fresh = |r: &Record| now.saturating_duration_since(r.seen) < REMEMBERED_FOR;
        if by_request.len() >= MOST_REMEMBERED {
            by_request.retain(|_, r| fresh(r));
        }
        let senders = by_request.values().filter(|r| r.sent_by == *sender).count();
        if senders >= MOST_A_SENDER {
            give_up_least_recently_seen(&mut by_request, |r| r.sent_by == *sender);
        }
        if by_request.len() >= MOST_REMEMBERED {
            give_up_least_recently_seen(&mut by_request, |_| true);
        }
        by_request.insert(
            request_id,
            Record {
                owner: coordinator,
                sent_by: sender.clone(),
                seen: now,
            },
        );
    }

    /// What `sender` is to `request_id`, at `now`.
    pub(crate) fn cancel_from(
        &self,
        request_id: &uuid::Uuid,
        sender: &NodeId,
        now: Instant,
    ) -> CancelFrom {
        let by_request = self.by_request.lock();
        match by_request.get(request_id) {
            Some(r) if now.saturating_duration_since(r.seen) < REMEMBERED_FOR => {
                if r.owner == *sender {
                    CancelFrom::Owner
                } else {
                    CancelFrom::NotTheOwner
                }
            }
            _ => CancelFrom::Unknown,
        }
    }

    /// Forget a conversation once its owner ended it.
    pub(crate) fn forget(&self, request_id: &uuid::Uuid) {
        self.by_request.lock().remove(request_id);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.by_request.lock().len()
    }
}

fn give_up_least_recently_seen(
    by_request: &mut HashMap<uuid::Uuid, Record>,
    among: impl Fn(&Record) -> bool,
) {
    let oldest = by_request
        .iter()
        .filter(|(_, r)| among(r))
        .min_by_key(|(_, r)| r.seen)
        .map(|(id, _)| *id);
    if let Some(id) = oldest {
        by_request.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    #[test]
    fn only_the_coordinator_the_first_forward_named_may_cancel() {
        let owners = RequestOwners::default();
        let r = uuid::Uuid::new_v4();
        let t = Instant::now();
        assert_eq!(owners.cancel_from(&r, &node(1), t), CancelFrom::Unknown);
        owners.note_forward(r, node(1), &node(1), t);
        assert_eq!(owners.cancel_from(&r, &node(1), t), CancelFrom::Owner);
        // Another computer of the chain learned the id: its cancel is not the owner's.
        assert_eq!(owners.cancel_from(&r, &node(2), t), CancelFrom::NotTheOwner);
        // A later forward naming another coordinator does not move the owner.
        owners.note_forward(r, node(2), &node(2), t);
        assert_eq!(owners.cancel_from(&r, &node(2), t), CancelFrom::NotTheOwner);
        assert_eq!(owners.cancel_from(&r, &node(1), t), CancelFrom::Owner);
        owners.forget(&r);
        assert_eq!(owners.cancel_from(&r, &node(1), t), CancelFrom::Unknown);
    }

    #[test]
    fn a_conversation_quiet_past_the_window_is_forgotten() {
        let owners = RequestOwners::default();
        let r = uuid::Uuid::new_v4();
        let t0 = Instant::now();
        owners.note_forward(r, node(1), &node(1), t0);
        let just_inside = t0 + REMEMBERED_FOR.saturating_sub(Duration::from_secs(1));
        assert_eq!(
            owners.cancel_from(&r, &node(1), just_inside),
            CancelFrom::Owner
        );
        let past = t0 + REMEMBERED_FOR + Duration::from_secs(1);
        assert_eq!(owners.cancel_from(&r, &node(1), past), CancelFrom::Unknown);
    }

    /// The review's attack: a peer opening ids of its own by the thousand
    /// evicted a live conversation's record, then claimed it. Now it only
    /// gives up its own.
    #[test]
    fn a_peer_opening_ids_by_the_thousand_pushes_out_only_its_own() {
        let owners = RequestOwners::default();
        let t = Instant::now();
        let live = uuid::Uuid::new_v4();
        owners.note_forward(live, node(1), &node(1), t);
        for i in 0..(MOST_REMEMBERED as u32 * 2) {
            let later = t + Duration::from_millis(u64::from(i) + 1);
            owners.note_forward(uuid::Uuid::new_v4(), node(9), &node(9), later);
        }
        assert_eq!(owners.len(), 1 + MOST_A_SENDER);
        let after = t + Duration::from_secs(60);
        assert_eq!(
            owners.cancel_from(&live, &node(1), after),
            CancelFrom::Owner
        );
        // And it cannot take the live one over.
        owners.note_forward(live, node(9), &node(9), after);
        assert_eq!(
            owners.cancel_from(&live, &node(9), after),
            CancelFrom::NotTheOwner
        );
    }

    /// A full table gives up the record seen least recently, never a live
    /// conversation's whose forwards keep it fresh.
    #[test]
    fn a_full_table_gives_up_the_least_recently_seen() {
        let owners = RequestOwners::default();
        let t = Instant::now();
        let live = uuid::Uuid::new_v4();
        owners.note_forward(live, node(0), &node(0), t);
        let mut n = 0u32;
        'fill: for sender in 1..=u8::MAX {
            for _ in 0..MOST_A_SENDER {
                if owners.len() >= MOST_REMEMBERED {
                    break 'fill;
                }
                n += 1;
                owners.note_forward(
                    uuid::Uuid::new_v4(),
                    node(sender),
                    &node(sender),
                    t + Duration::from_millis(u64::from(n)),
                );
            }
        }
        // The live conversation's next forward keeps it fresh.
        let fresh = t + Duration::from_secs(10);
        owners.note_forward(live, node(0), &node(0), fresh);
        owners.note_forward(uuid::Uuid::new_v4(), node(0), &node(0), fresh);
        assert_eq!(owners.len(), MOST_REMEMBERED);
        assert_eq!(
            owners.cancel_from(&live, &node(0), fresh),
            CancelFrom::Owner
        );
    }
}
