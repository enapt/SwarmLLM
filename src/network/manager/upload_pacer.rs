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
//! review of v0.3.252).

use std::time::{Duration, Instant};

/// The longest stretch of the upload limit one answer holds. At the default
/// 10 Mbps that is 5 MB of an 8 MiB chunk; at 100 Mbps the whole chunk.
pub(super) const ANSWER_HOLDS_AT_MOST: Duration = Duration::from_secs(4);

/// No answer is released later than this after its request: two thirds of the
/// downloader's `SHARD_STALL_SECS`, leaving the rest for the bytes to cross.
pub(super) const RELEASED_WITHIN: Duration = Duration::from_secs(super::SHARD_STALL_SECS * 2 / 3);

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
    /// The limit is spoken for past [`RELEASED_WITHIN`]: answer busy. An
    /// answer asked for `retry_after` from now would be admitted.
    Busy { retry_after: Duration },
}

/// The one timeline every served part's share of the upload limit is booked on.
#[derive(Debug, Default)]
pub(super) struct UploadPacer {
    /// When the last booked share ends; `None` before the first.
    next_free: Option<Instant>,
}

impl UploadPacer {
    /// Book a share of the limit for an answer asked to carry `asked` bytes,
    /// at `mbps` (read live by the caller; 0 = no limit, served at once).
    pub(super) fn reserve(&mut self, now: Instant, asked: u64, mbps: u64) -> Grant {
        if mbps == 0 {
            return Grant::Serve {
                bytes: asked,
                release: now,
            };
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
        if waits > RELEASED_WITHIN {
            return Grant::Busy {
                // Never saturates: this branch is `waits > RELEASED_WITHIN`.
                retry_after: waits
                    .saturating_sub(RELEASED_WITHIN)
                    .max(Duration::from_secs(1)),
            };
        }
        self.next_free = Some(release);
        Grant::Serve { bytes, release }
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
        let (b1, r1) = served(pacer.reserve(now, CHUNK, 10));
        let (b2, r2) = served(pacer.reserve(now, CHUNK, 10));
        let (b3, r3) = served(pacer.reserve(now, CHUNK, 10));
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
            served(pacer.reserve(now, CHUNK, 10)); // releases at 4, 8, … 20 s
        }
        match pacer.reserve(now, CHUNK, 10) {
            Grant::Busy { retry_after } => {
                assert_eq!(retry_after, Duration::from_secs(4), "24 s − 20 s");
            }
            g => panic!("expected busy, got {g:?}"),
        }
        // Nothing was booked by the refusal: once the queue has drained by
        // the wait it named, the same request is served.
        let later = now + Duration::from_secs(4);
        let (_, release) = served(pacer.reserve(later, CHUNK, 10));
        assert_eq!(release.duration_since(now), Duration::from_secs(24));
        assert!(RELEASED_WITHIN < Duration::from_secs(super::super::SHARD_STALL_SECS));
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
            served(pacer.reserve(now, asked, 10));
        }
        let (bytes, release) = served(pacer.reserve(now, bytes_to_book(part_len, 0, CHUNK), 10));
        assert_eq!(bytes, 5_000_000, "a real request still gets its full share");
        assert_eq!(release.duration_since(now), Duration::from_secs(4));
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
        let (bytes, release) = served(UploadPacer::default().reserve(now, CHUNK, 100));
        assert_eq!(bytes, CHUNK, "50 MB a share at 100 Mbps: the chunk whole");
        assert!(release.duration_since(now) < Duration::from_secs(1));
        assert_eq!(
            UploadPacer::default().reserve(now, CHUNK, 0),
            Grant::Serve {
                bytes: CHUNK,
                release: now
            }
        );
        let (bytes, release) = served(UploadPacer::default().reserve(now, CHUNK, 1));
        assert_eq!(bytes, 500_000, "4 s at 1 Mbps");
        assert_eq!(release.duration_since(now), Duration::from_secs(4));
        // A share that ended in the past does not hold the next one back.
        let mut pacer = UploadPacer::default();
        served(pacer.reserve(now, CHUNK, 10));
        let (_, release) = served(pacer.reserve(now + Duration::from_secs(60), CHUNK, 10));
        assert_eq!(
            release.duration_since(now + Duration::from_secs(60)),
            Duration::from_secs(4)
        );
    }
}
