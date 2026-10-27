//! One upload limit for every part this node serves (#335).
//!
//! `ResourcesConfig::shard_upload_mbps` exists so that seeding cannot fill
//! the owner's connection. It used to be applied per REQUEST: each answer
//! slept `bytes / limit` in its own task, so N parts served at once went up at
//! N times the limit (three at once moved at 3 × 1.25 MB/s between two rig
//! nodes, 2026-10-24). libtorrent and Transmission meter upload with ONE
//! session-wide limiter, and so does this: every answer reserves its share of
//! the limit on one shared timeline, and is released when its share ends.
//!
//! A shared limit queues, and a queue needs a bound. A downloader declares a
//! transfer stalled when no chunk has come for `SHARD_STALL_SECS` (30 s) and
//! starts the part again elsewhere, so:
//! - an answer holds at most [`ANSWER_HOLDS_AT_MOST`] of the limit — fewer
//!   bytes than asked where the limit is low. A downloader walks the part by
//!   the bytes it actually received (`new_offset = offset + chunk_len`), every
//!   build since chunking began, so a shorter answer needs no wire change;
//! - no answer is released later than [`RELEASED_WITHIN`] after its request.
//!   Past that the request is answered BUSY (`ShardResponse::busy`, to a
//!   downloader advertising `features::SHARD_BUSY`; empty to an older one).
//!
//! The reservation is made before the disk read, for the bytes the part HAS
//! from the offset asked ([`bytes_to_book`]). Booked for the bytes asked, a
//! request past a part's end held 4 s of the limit and sent nothing: five of
//! them, from any peer, answered every downloader busy for 20 s (pre-release
//! review of v0.3.252). An answer of no bytes goes at once: it holds no share,
//! so it does not wait in the inbound queue either.
//!
//! One timeline served first come, first served lets one downloader keep it
//! full, asking again the moment it is answered, while every other downloader
//! is answered busy (#340). So no NETWORK holds more than
//! [`NETWORK_HOLDS_AT_MOST`] of it — what a node fetching parts at the default
//! concurrency books, 12 of [`RELEASED_WITHIN`]'s 20 s — and the rest is always
//! there for the others. A network, not a key: a key costs nothing
//! to make, and this project counts every share a peer could multiply that way
//! by network (`network::helpers::NetGroup`, #299, #339). libtorrent shares its
//! one limit among peers too (`bandwidth_manager` hands each queued peer its
//! part of every tick's quota), and serves only a few at a time (unchoke
//! slots). A lone downloader loses nothing by it: at the default it is never
//! bounded, and asking for more it still has three answers queued, so the link
//! never runs dry while it asks again.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::network::helpers::NetGroup;

/// The longest stretch of the upload limit one answer holds. At the default
/// 10 Mbps that is 5 MB of an 8 MiB chunk; at 100 Mbps the whole chunk.
pub(super) const ANSWER_HOLDS_AT_MOST: Duration = Duration::from_secs(4);

/// No answer is released later than this after its request: two thirds of the
/// downloader's `SHARD_STALL_SECS`, leaving the rest for the bytes to cross.
pub(super) const RELEASED_WITHIN: Duration = Duration::from_secs(super::SHARD_STALL_SECS * 2 / 3);

/// The most of the timeline one network's answers hold at once (#340): the
/// answers of a node fetching as many parts at once as nodes do by default
/// (`DEFAULT_MAX_CONCURRENT_DOWNLOADS` × [`ANSWER_HOLDS_AT_MOST`], 12 s), so
/// such a node is never answered busy for its share. Half the window (10 s)
/// did: its third part waited every round and raced its siblings for the next
/// free share, and in the rig one part lost 16 times in a minute and left the
/// holder (2026-10-27). The rest of the window is there for everyone else.
pub(super) const NETWORK_HOLDS_AT_MOST: Duration = Duration::from_secs(
    ANSWER_HOLDS_AT_MOST.as_secs() * crate::config::DEFAULT_MAX_CONCURRENT_DOWNLOADS as u64,
);

// One answer always fits a network holding nothing, or a network alone could
// be answered busy for ever; and whoever else asks finds two answers' room.
const _: () = assert!(ANSWER_HOLDS_AT_MOST.as_secs() < NETWORK_HOLDS_AT_MOST.as_secs());
const _: () = assert!(
    NETWORK_HOLDS_AT_MOST.as_secs() + 2 * ANSWER_HOLDS_AT_MOST.as_secs()
        <= RELEASED_WITHIN.as_secs()
);

/// `SWARMLLM_UPLOAD_SHARE=0|off|false`: no network's share is bounded (before
/// #340), for an A/B inside one binary. Read once.
pub(super) fn network_share_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_UPLOAD_SHARE").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// What an answer to a part request may book for: the bytes asked, at most one
/// chunk (`SHARD_CHUNK_SIZE`), and at most what the part has from `offset` —
/// the same bound the read applies (`shard_transfer::read_shard_chunk_async`),
/// so a share is never booked for bytes that will not be sent.
pub(super) fn bytes_to_book(part_len: u64, offset: u64, asked: u64) -> u64 {
    asked
        .min(crate::network::protocol::SHARD_CHUNK_SIZE)
        .min(part_len.saturating_sub(offset))
}

/// What a part request may be given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Grant {
    /// Serve `bytes` (never more than asked) and send them at `release`.
    Serve { bytes: u64, release: Instant },
    /// The limit is spoken for past [`RELEASED_WITHIN`], or this network
    /// already holds its share ([`NETWORK_HOLDS_AT_MOST`]; `network_share`):
    /// answer busy. An answer asked for `retry_after` from now would be
    /// admitted.
    Busy {
        retry_after: Duration,
        network_share: bool,
    },
}

/// The one timeline every served part's share of the upload limit is booked on.
#[derive(Debug, Default)]
pub(super) struct UploadPacer {
    /// When the last booked share ends; `None` before the first.
    next_free: Option<Instant>,
    /// Every share not yet released, oldest first: the timeline only grows at
    /// its end, so booking order is release order. At most one per answer
    /// waiting to be sent, and those are bounded by the inbound queue.
    booked: VecDeque<Booked>,
}

/// One answer's share of the timeline.
#[derive(Debug, Clone, Copy)]
struct Booked {
    /// The network its request came from; `None` when shares are not bounded.
    from: Option<NetGroup>,
    release: Instant,
    hold: Duration,
}

impl Booked {
    /// What is left of this share at `now`.
    fn left(&self, now: Instant) -> Duration {
        self.hold.min(self.release.saturating_duration_since(now))
    }
}

/// The shortest wait a busy answer names.
const BUSY_WAIT_LEAST: Duration = Duration::from_secs(1);

impl UploadPacer {
    /// Book a share of the limit for an answer asked to carry `asked` bytes,
    /// at `mbps` (read live by the caller; 0 = no limit, served at once), for a
    /// request from the network `from` (`None`: shares not bounded, the A/B
    /// [`network_share_on`]).
    pub(super) fn reserve(
        &mut self,
        now: Instant,
        asked: u64,
        mbps: u64,
        from: Option<NetGroup>,
    ) -> Grant {
        if mbps == 0 || asked == 0 {
            return Grant::Serve {
                bytes: asked,
                release: now,
            };
        }
        while self.booked.front().is_some_and(|b| b.release <= now) {
            self.booked.pop_front();
        }
        let bytes_per_sec = mbps.saturating_mul(125_000);
        // The limit is a whole number of Mbps, so a share is never under
        // 500 KB (4 s at 1 Mbps): no floor is needed.
        let most = bytes_per_sec.saturating_mul(ANSWER_HOLDS_AT_MOST.as_secs());
        let bytes = asked.min(most);
        let hold = Duration::from_secs_f64(bytes as f64 / bytes_per_sec as f64);
        let start = self.next_free.filter(|t| *t > now).unwrap_or(now);
        let release = start + hold;
        let waits = release.duration_since(now);
        let queue_waits = (waits > RELEASED_WITHIN).then(|| waits.saturating_sub(RELEASED_WITHIN));
        let network_waits = from.and_then(|from| self.network_waits(now, from, hold));
        if let Some(wait) = queue_waits.max(network_waits) {
            return Grant::Busy {
                retry_after: wait.max(BUSY_WAIT_LEAST),
                network_share: network_waits.is_some_and(|w| Some(w) >= queue_waits),
            };
        }
        self.next_free = Some(release);
        self.booked.push_back(Booked {
            from,
            release,
            hold,
        });
        Grant::Serve { bytes, release }
    }

    /// How long until `from` may hold `hold` more of the timeline, if it may
    /// not now: what is left of its shares, and `hold`, may not pass
    /// [`NETWORK_HOLDS_AT_MOST`]. Its shares end oldest first, and one that has
    /// not begun waits for the one before it to end, so the wait is to the end
    /// of the first share whose end leaves enough room.
    fn network_waits(&self, now: Instant, from: NetGroup, hold: Duration) -> Option<Duration> {
        let its = || self.booked.iter().filter(move |b| b.from == Some(from));
        let held: Duration = its().map(|b| b.left(now)).sum();
        let mut over = (held + hold).checked_sub(NETWORK_HOLDS_AT_MOST)?;
        if over.is_zero() {
            return None;
        }
        for share in its() {
            let left = share.left(now);
            if left >= over {
                return Some(share.release.saturating_duration_since(now));
            }
            over -= left;
        }
        // `hold` alone is past the bound: never, as an answer holds at most
        // ANSWER_HOLDS_AT_MOST.
        Some(BUSY_WAIT_LEAST)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK: u64 = crate::network::protocol::SHARD_CHUNK_SIZE;

    fn served(g: Grant) -> (u64, Instant) {
        match g {
            Grant::Serve { bytes, release } => (bytes, release),
            Grant::Busy { .. } => panic!("expected a share, got {g:?}"),
        }
    }

    /// The defect: three parts served at once each slept its own `bytes /
    /// limit` and went up together at three times the limit. Booked on one
    /// timeline, each answer starts where the last one's share ends, so
    /// together they never pass the limit.
    #[test]
    fn parts_served_at_once_share_one_limit() {
        let mut pacer = UploadPacer::default();
        let now = Instant::now();
        let (b1, r1) = served(pacer.reserve(now, CHUNK, 10, None));
        let (b2, r2) = served(pacer.reserve(now, CHUNK, 10, None));
        let (b3, r3) = served(pacer.reserve(now, CHUNK, 10, None));
        // 10 Mbps = 1.25 MB/s: an answer holds 4 s, 5 MB of the 8 MiB asked.
        assert_eq!(b1, 5_000_000);
        assert_eq!((b2, b3), (b1, b1));
        let secs = |r: Instant| r.duration_since(now).as_secs_f64();
        assert!((secs(r1) - 4.0).abs() < 0.01, "{}", secs(r1));
        assert!((secs(r2) - 8.0).abs() < 0.01, "{}", secs(r2));
        assert!((secs(r3) - 12.0).abs() < 0.01, "{}", secs(r3));
        // Bytes out over the time they took: the limit, not three times it.
        let rate = (b1 + b2 + b3) as f64 / secs(r3);
        assert!((rate - 1_250_000.0).abs() < 1_000.0, "{rate}");
    }

    /// Past what can be released within the downloader's patience, the
    /// answer is busy — with a wait after which it would be admitted — and a
    /// busy answer books nothing.
    #[test]
    fn a_queue_past_its_bound_answers_busy_and_books_nothing() {
        let mut pacer = UploadPacer::default();
        let now = Instant::now();
        for _ in 0..5 {
            served(pacer.reserve(now, CHUNK, 10, None)); // releases at 4, 8, … 20 s
        }
        match pacer.reserve(now, CHUNK, 10, None) {
            Grant::Busy {
                retry_after,
                network_share,
            } => {
                assert_eq!(retry_after, Duration::from_secs(4), "24 s − 20 s");
                assert!(!network_share);
            }
            g => panic!("expected busy, got {g:?}"),
        }
        // Nothing was booked by the refusal: once the queue has drained by
        // the wait it named, the same request is served.
        let later = now + Duration::from_secs(4);
        let (_, release) = served(pacer.reserve(later, CHUNK, 10, None));
        assert_eq!(release.duration_since(now), Duration::from_secs(24));
        assert!(RELEASED_WITHIN < Duration::from_secs(super::super::SHARD_STALL_SECS));
    }

    fn network(last: u8) -> NetGroup {
        NetGroup::Direct(crate::network::helpers::Network::Host(
            std::net::IpAddr::from([10, 0, 0, last]),
        ))
    }

    /// The defect (#340): one downloader asking again as fast as it is
    /// answered filled the timeline, and everyone else was answered busy. Now
    /// a network holds at most its share of it, and another network is served.
    #[test]
    fn one_network_never_holds_more_than_its_share_of_the_window() {
        let now = Instant::now();
        let (greedy, other) = (network(1), network(2));

        // Unbounded (the A/B arm, and before #340): five answers to one
        // network fill the 20 s window; the other network is turned away.
        let mut pacer = UploadPacer::default();
        for _ in 0..5 {
            served(pacer.reserve(now, CHUNK, 10, None));
        }
        assert!(matches!(
            pacer.reserve(now, CHUNK, 10, Some(other)),
            Grant::Busy {
                network_share: false,
                ..
            }
        ));

        // Bounded: three 4 s answers are 12 s; a fourth would be 16 s.
        let mut pacer = UploadPacer::default();
        for _ in 0..3 {
            served(pacer.reserve(now, CHUNK, 10, Some(greedy)));
        }
        match pacer.reserve(now, CHUNK, 10, Some(greedy)) {
            Grant::Busy {
                retry_after,
                network_share,
            } => {
                assert_eq!(
                    retry_after,
                    Duration::from_secs(4),
                    "when its first share ends"
                );
                assert!(network_share);
            }
            g => panic!("expected busy, got {g:?}"),
        }
        // The other network is served, after what is already booked.
        let (_, release) = served(pacer.reserve(now, CHUNK, 10, Some(other)));
        assert_eq!(release.duration_since(now), Duration::from_secs(16));
        // At the wait it was named, the greedy network is served again.
        let later = now + Duration::from_secs(4);
        let (_, release) = served(pacer.reserve(later, CHUNK, 10, Some(greedy)));
        assert_eq!(release.duration_since(now), Duration::from_secs(20));
        assert_eq!(NETWORK_HOLDS_AT_MOST, Duration::from_secs(12));
    }

    /// A node fetching as many parts at once as nodes do by default is never
    /// answered busy for its share: each part asks again when its answer
    /// leaves, and the three always fit. With half the window as the bound
    /// (10 s), the third part was turned away every round, raced the others
    /// for the next free share and, in the rig, lost 16 times in a minute and
    /// left the holder (2026-10-27).
    #[test]
    fn a_node_fetching_at_the_default_concurrency_is_never_answered_busy() {
        let now = Instant::now();
        let alone = network(1);
        let mut pacer = UploadPacer::default();
        let mut asks: Vec<Instant> = vec![now; crate::config::DEFAULT_MAX_CONCURRENT_DOWNLOADS];
        for _ in 0..12 {
            for at in asks.iter_mut() {
                match pacer.reserve(*at, CHUNK, 10, Some(alone)) {
                    Grant::Serve { release, .. } => *at = release,
                    g => panic!("a part at the default concurrency was answered {g:?}"),
                }
            }
        }
    }

    /// A downloader alone loses nothing: by the time the bound lets it book
    /// again, the link is still busy with what it booked before, so its shares
    /// run back to back at the full limit.
    #[test]
    fn a_lone_network_keeps_the_link_busy() {
        let now = Instant::now();
        let alone = network(1);
        let mut pacer = UploadPacer::default();
        let mut releases = Vec::new();
        let mut at = now;
        // Three parts at once, each asking again the moment it may.
        while releases.len() < 12 {
            match pacer.reserve(at, CHUNK, 10, Some(alone)) {
                Grant::Serve { release, .. } => releases.push(release),
                Grant::Busy { retry_after, .. } => at += retry_after,
            }
        }
        for pair in releases.windows(2) {
            assert_eq!(pair[1] - pair[0], Duration::from_secs(4), "{releases:?}");
        }
        let rate = 12.0 * 5_000_000.0 / releases[11].duration_since(now).as_secs_f64();
        assert!((rate - 1_250_000.0).abs() < 1_000.0, "{rate}");
    }

    /// A request for bytes the part does not have books nothing: a peer asking
    /// past a part's end, over and over, cannot hold the limit other
    /// downloaders share.
    #[test]
    fn a_request_past_a_parts_end_books_nothing() {
        let mut pacer = UploadPacer::default();
        let now = Instant::now();
        let part_len = 100 * 1024 * 1024;
        assert_eq!(bytes_to_book(part_len, u64::MAX, CHUNK), 0);
        assert_eq!(bytes_to_book(part_len, part_len, CHUNK), 0);
        for _ in 0..50 {
            let asked = bytes_to_book(part_len, u64::MAX, CHUNK);
            served(pacer.reserve(now, asked, 10, None));
        }
        let (bytes, release) =
            served(pacer.reserve(now, bytes_to_book(part_len, 0, CHUNK), 10, None));
        assert_eq!(bytes, 5_000_000, "a real request still gets its full share");
        assert_eq!(release.duration_since(now), Duration::from_secs(4));
        // An answer of no bytes goes at once, behind no share — so it does not
        // wait in the inbound queue either.
        assert_eq!(
            pacer.reserve(now, 0, 10, None),
            Grant::Serve {
                bytes: 0,
                release: now
            }
        );
        // The last chunk books only what is left; a request is never booked
        // past one chunk.
        assert_eq!(bytes_to_book(part_len, part_len - 10, CHUNK), 10);
        assert_eq!(bytes_to_book(part_len, 0, u64::MAX), CHUNK);
    }

    /// A high limit serves the whole chunk asked for; no limit is no wait; the
    /// lowest limit still sends 500 KB an answer; a share that ended in the
    /// past holds nothing back.
    #[test]
    fn an_answer_is_cut_only_by_a_low_limit() {
        let now = Instant::now();
        let (bytes, release) = served(UploadPacer::default().reserve(now, CHUNK, 100, None));
        assert_eq!(bytes, CHUNK, "50 MB a share at 100 Mbps: the chunk whole");
        assert!(release.duration_since(now) < Duration::from_secs(1));
        assert_eq!(
            UploadPacer::default().reserve(now, CHUNK, 0, None),
            Grant::Serve {
                bytes: CHUNK,
                release: now
            }
        );
        let (bytes, release) = served(UploadPacer::default().reserve(now, CHUNK, 1, None));
        assert_eq!(bytes, 500_000, "4 s at 1 Mbps");
        assert_eq!(release.duration_since(now), Duration::from_secs(4));
        // A share that ended in the past does not hold the next one back.
        let mut pacer = UploadPacer::default();
        served(pacer.reserve(now, CHUNK, 10, None));
        let (_, release) = served(pacer.reserve(now + Duration::from_secs(60), CHUNK, 10, None));
        assert_eq!(
            release.duration_since(now + Duration::from_secs(60)),
            Duration::from_secs(4)
        );
    }
}
