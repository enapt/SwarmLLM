//! What each computer and this one have done for each other, as THIS node saw
//! it — the per-peer give and take shown in the peer list.
//!
//! Every node already minted its own credit balance, so a balance says nothing
//! (`docs/CREDITS_DESIGN.md` § 1: a freeloader's number reads as high as anyone's).
//! What does show a computer that only takes is a count each side keeps of the
//! other: the design's § 3.2 bilateral tab, here unsigned and gating nothing.
//! It is the shape BitTorrent clients have shown for twenty years — libtorrent's
//! `peer_info::total_upload` / `total_download`, qBittorrent's per-peer
//! "Uploaded" / "Downloaded" columns — and the reason is the same one tit-for-tat
//! rests on (Cohen 2003, "Incentives Build Robustness in BitTorrent"): a node
//! trusts only what it observed itself, so no third party can inflate a count
//! or blame anyone.
//!
//! Both directions are counted in whole requests, and only from what the
//! network authenticated:
//! - **ran for them**: a request this node took on for the peer that SENT it —
//!   a whole hand-off, a split it leads, or the first forward of a peer's split
//!   through a part here (`SharedState::note_request_served_for_a_peer`). A
//!   chained forward that names a coordinator other than its sender is not
//!   counted: the name is not authenticated, and counting it would let anyone
//!   make another computer look like a taker. The first hop of that chain,
//!   which the coordinator asked directly, counts it.
//! - **ran for us**: a request this node sent the peer that it saw through to
//!   the end — a segment of a split led here, or a whole hand-off
//!   (`SharedState::note_peer_completed_request`).
//!
//! ⚠ A description of work, not a balance. Credits are DORMANT; nothing may
//! read a tab to decide whom to serve.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

use crate::types::NodeId;

/// The redb tree and key the tabs live under.
const TREE: &str = "peer_exchange";
const KEY: &str = "tabs";

/// The most computers remembered. A computer seen least recently makes room
/// for a new one, so a peer churning identities cannot grow the map.
pub const MAX_TABS: usize = 1024;

/// One computer's tab, both directions in whole requests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tab {
    /// Requests this node took on for the computer.
    #[serde(default)]
    pub ran_for_them: u64,
    /// Requests the computer saw through for this node.
    #[serde(default)]
    pub ran_for_us: u64,
    /// When either count last moved, epoch millis — what eviction goes by.
    #[serde(default)]
    pub last_ms: u64,
}

/// A tab as stored: the node id in hex, so the record reads as text.
#[derive(Serialize, Deserialize)]
struct StoredTab {
    node: String,
    tab: Tab,
}

/// The most recent (request, computer, direction) counts remembered, so one
/// request is counted once however many paths report it — a whole hand-off is
/// seen through by the hand-off AND by the router's success loop, and a peer
/// re-planning one request sends its first forward again.
const RECENT_COUNTED: usize = 1024;

/// Every computer's tab, and how far the record on disk has caught up.
pub struct PeerExchange {
    tabs: DashMap<NodeId, Tab>,
    /// Bumped by every change; the record is written when it moved.
    changes: AtomicU64,
    written_at: parking_lot::Mutex<u64>,
    /// What was counted lately, newest last (`RECENT_COUNTED`): `true` = ran
    /// for us. Memory only — a restart forgets which request was counted, not
    /// the counts.
    recent: parking_lot::Mutex<std::collections::VecDeque<(uuid::Uuid, NodeId, bool)>>,
}

impl PeerExchange {
    /// Read the tabs a previous run left. A record that cannot be read is a
    /// fresh start, said once — never a reason not to run.
    pub fn load(db: &crate::storage::db::Database) -> Self {
        let exchange = Self::empty();
        match db.get_json::<Vec<StoredTab>>(TREE, KEY) {
            Ok(Some(stored)) => {
                let mut tabs: Vec<(NodeId, Tab)> = stored
                    .into_iter()
                    .filter_map(|s| {
                        let bytes: [u8; 32] = hex::decode(&s.node).ok()?.try_into().ok()?;
                        Some((NodeId(bytes), s.tab))
                    })
                    .collect();
                tabs.sort_by_key(|(_, tab)| std::cmp::Reverse(tab.last_ms));
                tabs.truncate(MAX_TABS);
                for (node, tab) in tabs {
                    exchange.tabs.insert(node, tab);
                }
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!(error = %e, "Could not read what peers and this computer did for each other — starting afresh")
            }
        }
        exchange
    }

    fn empty() -> Self {
        Self {
            tabs: DashMap::new(),
            changes: AtomicU64::new(0),
            written_at: parking_lot::Mutex::new(0),
            recent: parking_lot::Mutex::new(std::collections::VecDeque::new()),
        }
    }

    /// `node`'s tab; all zeros for a computer this node has exchanged nothing with.
    pub fn tab_of(&self, node: &NodeId) -> Tab {
        self.tabs.get(node).map(|t| *t).unwrap_or_default()
    }

    /// This node took on `request` for `node`. Written only by
    /// `SharedState::note_request_served_for_a_peer`.
    pub(super) fn note_ran_for_them(&self, request: uuid::Uuid, node: &NodeId) {
        if self.first_count_of(request, node, false) {
            self.note(node, |tab| {
                tab.ran_for_them = tab.ran_for_them.saturating_add(1)
            });
        }
    }

    /// `node` saw `request` of this node's through. Written only by
    /// `SharedState::note_peer_completed_request`.
    pub(super) fn note_ran_for_us(&self, request: uuid::Uuid, node: &NodeId) {
        if self.first_count_of(request, node, true) {
            self.note(node, |tab| {
                tab.ran_for_us = tab.ran_for_us.saturating_add(1)
            });
        }
    }

    /// Is this the first time `request` is counted for `node` this way? Remembers it.
    fn first_count_of(&self, request: uuid::Uuid, node: &NodeId, for_us: bool) -> bool {
        let mut recent = self.recent.lock();
        if recent
            .iter()
            .any(|(r, n, d)| *r == request && n == node && *d == for_us)
        {
            return false;
        }
        if recent.len() >= RECENT_COUNTED {
            recent.pop_front();
        }
        recent.push_back((request, node.clone(), for_us));
        true
    }

    fn note(&self, node: &NodeId, change: impl FnOnce(&mut Tab)) {
        if !self.tabs.contains_key(node) && self.tabs.len() >= MAX_TABS {
            self.forget_the_stalest();
        }
        let now_ms = u64::try_from(chrono::Utc::now().timestamp_millis()).unwrap_or(0);
        let mut tab = self.tabs.entry(node.clone()).or_default();
        change(&mut tab);
        tab.last_ms = now_ms;
        drop(tab);
        self.changes.fetch_add(1, Relaxed);
    }

    fn forget_the_stalest(&self) {
        let stalest = self
            .tabs
            .iter()
            .min_by_key(|entry| entry.value().last_ms)
            .map(|entry| entry.key().clone());
        if let Some(node) = stalest {
            self.tabs.remove(&node);
        }
    }

    fn stored(&self) -> Vec<StoredTab> {
        self.tabs
            .iter()
            .map(|entry| StoredTab {
                node: hex::encode(entry.key().0),
                tab: *entry.value(),
            })
            .collect()
    }
}

impl super::SharedState {
    /// Write the tabs if any moved since the last write. Called on the health
    /// monitor's tick and once at shutdown; an idle node writes nothing.
    pub fn persist_peer_exchange(&self) {
        let exchange = &self.metrics.peer_exchange;
        let changes = exchange.changes.load(Relaxed);
        let mut written_at = exchange.written_at.lock();
        if *written_at == changes {
            return;
        }
        match self.db.put_json(TREE, KEY, &exchange.stored()) {
            Ok(()) => *written_at = changes,
            Err(e) => {
                tracing::debug!(error = %e, "Could not write the peer give-and-take record — next tick retries")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(n: u8) -> NodeId {
        NodeId([n; 32])
    }

    /// **What it is for**: a restart keeps each computer's tab, both ways — an
    /// auto-update restarts a node about once a day, and a count that reset
    /// with it would show every freeloader as new.
    #[test]
    fn a_restart_keeps_each_computers_tab() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(dir.path()).unwrap();
        let first = PeerExchange::load(&db);
        first.note_ran_for_them(uuid::Uuid::new_v4(), &node(1));
        first.note_ran_for_them(uuid::Uuid::new_v4(), &node(1));
        first.note_ran_for_us(uuid::Uuid::new_v4(), &node(2));
        db.put_json(TREE, KEY, &first.stored()).unwrap();

        let second = PeerExchange::load(&db);
        assert_eq!(second.tab_of(&node(1)).ran_for_them, 2);
        assert_eq!(second.tab_of(&node(1)).ran_for_us, 0);
        assert_eq!(second.tab_of(&node(2)).ran_for_us, 1);
        assert_eq!(
            second.tab_of(&node(3)),
            Tab::default(),
            "a stranger reads zero"
        );
    }

    /// The map is bounded: past `MAX_TABS`, the computer heard from least
    /// recently makes room, so identities churned by one peer cannot grow it.
    #[test]
    fn the_least_recent_computer_makes_room() {
        let exchange = PeerExchange::empty();
        for i in 0..MAX_TABS {
            let mut id = [0u8; 32];
            id[..8].copy_from_slice(&(i as u64).to_le_bytes());
            exchange.tabs.insert(
                NodeId(id),
                Tab {
                    ran_for_them: 1,
                    ran_for_us: 0,
                    last_ms: 1_000 + i as u64,
                },
            );
        }
        let oldest = NodeId([0u8; 32]);
        exchange.note_ran_for_us(uuid::Uuid::new_v4(), &node(0xEE));
        assert_eq!(exchange.tabs.len(), MAX_TABS);
        assert!(!exchange.tabs.contains_key(&oldest), "the stalest went");
        assert_eq!(exchange.tab_of(&node(0xEE)).ran_for_us, 1);
    }

    /// One request counts once, however many paths report it: a whole
    /// hand-off is seen through by the hand-off and again by the router's
    /// success loop. A second request counts again.
    #[test]
    fn one_request_counts_once_however_many_paths_report_it() {
        let exchange = PeerExchange::empty();
        let request = uuid::Uuid::new_v4();
        exchange.note_ran_for_us(request, &node(1));
        exchange.note_ran_for_us(request, &node(1));
        exchange.note_ran_for_them(request, &node(1));
        exchange.note_ran_for_them(request, &node(1));
        assert_eq!(exchange.tab_of(&node(1)).ran_for_us, 1);
        assert_eq!(exchange.tab_of(&node(1)).ran_for_them, 1);
        exchange.note_ran_for_us(uuid::Uuid::new_v4(), &node(1));
        assert_eq!(exchange.tab_of(&node(1)).ran_for_us, 2);
    }
}
