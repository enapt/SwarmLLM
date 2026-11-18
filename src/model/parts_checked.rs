//! What this node has already checked of the parts it holds (FUTURE_WORK #477).
//!
//! A record says: the bytes at this part's path — this size, this
//! modification time, this file — hashed to this BLAKE3, at this time. While
//! all of it still holds, reading the part again can only give the same
//! answer, so [`super::shard::ShardStore::verify_shard`] (asked with
//! [`super::shard::Reread::IfChanged`]) takes the record instead of the read.
//! Before it, every start hashed every held part whole: 443 parts kept a
//! mechanical disk saturated for 10-15 minutes (field report, 2026-11-18).
//!
//! **Written** only where bytes were just hashed and matched: a passing
//! `verify_shard` and a HuggingFace part kept by `accept_origin_part`. A
//! mismatch, a dispute, a quarantine never writes one, and a record is written
//! only when the file's stamp was the same before the hash as after it — a
//! part changed while it was read is read again next time (git's "racy clean"
//! problem: a stat taken after a write in the same clock tick cannot tell the
//! write happened).
//!
//! **Not trusted** when the file's size, modification time or identity
//! (device and inode on Unix) differ — every writer here writes beside a part
//! and renames over it, which is a new inode; an edit in place moves the
//! modification time; when the hash it names is not the hash the caller
//! checks against now (a changed expectation re-reads, #224); or when it is
//! older than [`RECHECK_EVERY`] — the slow re-read that catches a disk
//! damaging bytes it never rewrote, which no stamp can see.
//!
//! Prior art: libtorrent's resume data keeps each file's size and modification
//! time and rechecks a torrent only where they disagree; git's index caches
//! (size, mtime, ctime, dev, ino) → blob hash and re-hashes a "racily clean"
//! entry; Btrfs (`btrfsmaintenance`) and Debian's md `checkarray` scrub
//! MONTHLY by default — the period taken here. ctime is left out (git's own
//! `core.trustctime` exists because backup and indexing tools touch it).
//! A/B: `SWARMLLM_PARTS_CHECKED=0` trusts no record (every check reads).

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::storage::db::Database;
use crate::types::ModelId;

/// The redb tree holding one [`CheckedPart`] per part this node checked.
/// Private: only [`PartsChecked`] writes or reads it.
const PARTS_CHECKED_TREE: &str = "parts_checked";

/// How long a check stands before the part is read again: 30 days, the
/// monthly scrub of btrfsmaintenance and Debian's md `checkarray`.
pub const RECHECK_EVERY: Duration = Duration::from_secs(30 * 24 * 3600);

/// A record stamped further than this into the future of this computer's
/// clock is not trusted: the clock was set back, and "checked in the future"
/// would otherwise stand until the clock caught up.
const AHEAD_OF_THE_CLOCK: Duration = Duration::from_secs(24 * 3600);

/// What a file's metadata says of WHICH bytes it holds — compared whole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStamp {
    pub size: u64,
    pub modified_secs: u64,
    pub modified_nanos: u32,
    /// Device and inode where the platform names them (Unix): a part replaced
    /// by a rename is a different file even within the clock's resolution.
    #[serde(default)]
    pub device: Option<u64>,
    #[serde(default)]
    pub inode: Option<u64>,
}

impl FileStamp {
    /// The stamp of the file at `path`; `None` when it cannot be read whole
    /// (no file, a modification time before 1970), which no record matches.
    pub fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        #[cfg(unix)]
        let (device, inode) = {
            use std::os::unix::fs::MetadataExt;
            (Some(meta.dev()), Some(meta.ino()))
        };
        #[cfg(not(unix))]
        let (device, inode) = (None, None);
        Some(Self {
            size: meta.len(),
            modified_secs: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
            device,
            inode,
        })
    }

    /// The stamp of bytes that were hashed, if the file was the same file
    /// before the hash (`before`) as after it (`after`).
    pub fn unchanged(before: Option<Self>, after: Option<Self>) -> Option<Self> {
        match (before, after) {
            (Some(b), Some(a)) if a == b => Some(b),
            _ => None,
        }
    }
}

/// One part's record (the row's value).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckedPart {
    model_id: String,
    index: u32,
    stamp: FileStamp,
    /// The BLAKE3 the bytes hashed to, hex.
    hash: String,
    checked_at_ms: u64,
}

/// The records, over the node's database. Cheap to clone.
#[derive(Clone)]
pub struct PartsChecked {
    db: Database,
    trusted: bool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn key(model_id: &ModelId, index: u32) -> String {
    format!("{}/{index}", model_id.0)
}

impl PartsChecked {
    pub fn new(db: Database) -> Self {
        static TRUSTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let trusted = *TRUSTED.get_or_init(|| {
            !std::env::var("SWARMLLM_PARTS_CHECKED").is_ok_and(|v| v.trim() == "0")
        });
        Self { db, trusted }
    }

    /// Whether a check of THESE bytes — `stamp` — found them hashing to
    /// `expected` within [`RECHECK_EVERY`].
    pub fn vouches_for(
        &self,
        model_id: &ModelId,
        index: u32,
        expected: &[u8; 32],
        stamp: &FileStamp,
    ) -> bool {
        self.trusted && self.vouches_for_at(model_id, index, expected, stamp, now_ms())
    }

    fn vouches_for_at(
        &self,
        model_id: &ModelId,
        index: u32,
        expected: &[u8; 32],
        stamp: &FileStamp,
        now_ms: u64,
    ) -> bool {
        let Ok(Some(rec)) = self
            .db
            .get_json::<CheckedPart>(PARTS_CHECKED_TREE, &key(model_id, index))
        else {
            return false;
        };
        let age_ms = now_ms.saturating_sub(rec.checked_at_ms);
        rec.stamp == *stamp
            && rec.hash == hex::encode(expected)
            && age_ms < RECHECK_EVERY.as_millis() as u64
            && rec.checked_at_ms <= now_ms.saturating_add(AHEAD_OF_THE_CLOCK.as_millis() as u64)
    }

    /// Record that the bytes stamped `stamp` hashed to `hash` just now.
    /// `None` (the file moved while it was read) records nothing. A failed
    /// write costs one more read later, so it is logged, not returned.
    pub fn note_checked(
        &self,
        model_id: &ModelId,
        index: u32,
        hash: &[u8; 32],
        stamp: Option<FileStamp>,
    ) {
        self.note_checked_at(model_id, index, hash, stamp, now_ms());
    }

    fn note_checked_at(
        &self,
        model_id: &ModelId,
        index: u32,
        hash: &[u8; 32],
        stamp: Option<FileStamp>,
        now_ms: u64,
    ) {
        let Some(stamp) = stamp else {
            tracing::debug!(
                model = %model_id,
                shard = index,
                "The part changed while it was checked — nothing recorded, it is read again next time"
            );
            return;
        };
        let rec = CheckedPart {
            model_id: model_id.0.clone(),
            index,
            stamp,
            hash: hex::encode(hash),
            checked_at_ms: now_ms,
        };
        if let Err(e) = self
            .db
            .put_json(PARTS_CHECKED_TREE, &key(model_id, index), &rec)
        {
            tracing::debug!(
                model = %model_id,
                shard = index,
                error = %e,
                "Could not record a part as checked — it is read again next time"
            );
        }
    }

    /// Drop every record whose file no longer is the file it stamped, as
    /// `stamp_now` reads it (`None` = gone) — a part given back, deleted or
    /// replaced. Run after each background pass, so the tree holds at most
    /// one row per part on disk. How many went.
    pub fn forget_stale(&self, stamp_now: impl Fn(&ModelId, u32) -> Option<FileStamp>) -> usize {
        // Judged inside the removing transaction: a record a check rewrote
        // after a snapshot would be judged on its OLD stamp and lost.
        match self
            .db
            .remove_json_where::<CheckedPart>(PARTS_CHECKED_TREE, |_, rec| {
                stamp_now(&ModelId(rec.model_id.clone()), rec.index) != Some(rec.stamp)
            }) {
            Ok(n) => n,
            Err(e) => {
                tracing::debug!(error = %e, "Could not forget stale part-check records");
                0
            }
        }
    }

    /// How many records there are.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.db
            .get_all_json::<CheckedPart>(PARTS_CHECKED_TREE)
            .map(|r| r.len())
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self {
            db: Database::open_temp().unwrap(),
            trusted: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(dir: &Path, bytes: &[u8]) -> std::path::PathBuf {
        let p = dir.join("shard_000.bin");
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn a_record_vouches_only_for_the_bytes_and_hash_it_names() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"hello part");
        let checks = PartsChecked::for_tests();
        let m = ModelId("m".into());
        let hash = *blake3::hash(b"hello part").as_bytes();
        let stamp = FileStamp::of(&path);
        assert!(stamp.is_some());
        assert!(!checks.vouches_for(&m, 0, &hash, &stamp.unwrap()));
        checks.note_checked(&m, 0, &hash, stamp);
        assert!(checks.vouches_for(&m, 0, &hash, &stamp.unwrap()));
        // Another hash expected now (#224): the record is no answer.
        assert!(!checks.vouches_for(&m, 0, &[7u8; 32], &stamp.unwrap()));
        // Another part, another model: no answer either.
        assert!(!checks.vouches_for(&m, 1, &hash, &stamp.unwrap()));
        assert!(!checks.vouches_for(&ModelId("n".into()), 0, &hash, &stamp.unwrap()));
    }

    #[test]
    fn a_part_rewritten_the_same_size_is_not_vouched_for() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"aaaaaaaaaa");
        let checks = PartsChecked::for_tests();
        let m = ModelId("m".into());
        let hash = *blake3::hash(b"aaaaaaaaaa").as_bytes();
        checks.note_checked(&m, 0, &hash, FileStamp::of(&path));
        // In place, same size, and a modification time a second later — what
        // `dd conv=notrunc` does to a part (the rigs' corruption).
        std::fs::write(&path, b"bbbbbbbbbb").unwrap();
        let later = SystemTime::now() + Duration::from_secs(1);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(later)
            .unwrap();
        assert!(!checks.vouches_for(&m, 0, &hash, &FileStamp::of(&path).unwrap()));
    }

    #[cfg(unix)]
    #[test]
    fn a_part_replaced_by_a_rename_is_another_file_whatever_its_time() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"aaaaaaaaaa");
        let checks = PartsChecked::for_tests();
        let m = ModelId("m".into());
        let hash = *blake3::hash(b"aaaaaaaaaa").as_bytes();
        let before = FileStamp::of(&path).unwrap();
        checks.note_checked(&m, 0, &hash, Some(before));
        // A writer's way: beside it, then renamed over it — with the very
        // same modification time, as a clock with coarse ticks could give.
        let tmp = dir.path().join("shard_000.bin.tmp");
        std::fs::write(&tmp, b"bbbbbbbbbb").unwrap();
        let t = UNIX_EPOCH + Duration::new(before.modified_secs, before.modified_nanos);
        std::fs::File::options()
            .write(true)
            .open(&tmp)
            .unwrap()
            .set_modified(t)
            .unwrap();
        std::fs::rename(&tmp, &path).unwrap();
        let after = FileStamp::of(&path).unwrap();
        assert_eq!(
            (after.size, after.modified_secs, after.modified_nanos),
            (before.size, before.modified_secs, before.modified_nanos)
        );
        assert!(!checks.vouches_for(&m, 0, &hash, &after));
    }

    #[test]
    fn a_check_stands_for_thirty_days_and_no_longer() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"x");
        let checks = PartsChecked::for_tests();
        let m = ModelId("m".into());
        let hash = *blake3::hash(b"x").as_bytes();
        let stamp = FileStamp::of(&path).unwrap();
        let t = 1_800_000_000_000u64;
        checks.note_checked_at(&m, 0, &hash, Some(stamp), t);
        let day = 24 * 3600 * 1000;
        assert!(checks.vouches_for_at(&m, 0, &hash, &stamp, t + 29 * day));
        assert!(!checks.vouches_for_at(&m, 0, &hash, &stamp, t + 30 * day));
        // A clock set back a little still trusts it; set back far, it does not.
        assert!(checks.vouches_for_at(&m, 0, &hash, &stamp, t - 3600 * 1000));
        assert!(!checks.vouches_for_at(&m, 0, &hash, &stamp, t - 2 * day));
    }

    #[test]
    fn a_part_that_moved_while_it_was_read_is_not_recorded() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"x");
        let checks = PartsChecked::for_tests();
        let m = ModelId("m".into());
        let hash = *blake3::hash(b"x").as_bytes();
        let before = FileStamp::of(&path);
        let mut moved = before.unwrap();
        moved.modified_nanos = moved.modified_nanos.wrapping_add(1);
        checks.note_checked(&m, 0, &hash, FileStamp::unchanged(before, Some(moved)));
        assert!(checks.is_empty());
        checks.note_checked(&m, 0, &hash, FileStamp::unchanged(before, before));
        assert_eq!(checks.len(), 1);
    }

    #[test]
    fn stale_records_are_forgotten_and_current_ones_kept() {
        let dir = tempfile::tempdir().unwrap();
        let path = part(dir.path(), b"x");
        let checks = PartsChecked::for_tests();
        let hash = *blake3::hash(b"x").as_bytes();
        let stamp = FileStamp::of(&path);
        checks.note_checked(&ModelId("kept".into()), 0, &hash, stamp);
        checks.note_checked(&ModelId("gone".into()), 0, &hash, stamp);
        let gone = checks.forget_stale(|m, _| (m.0 == "kept").then(|| stamp.unwrap()));
        assert_eq!(gone, 1);
        assert_eq!(checks.len(), 1);
        assert!(checks.vouches_for(&ModelId("kept".into()), 0, &hash, &stamp.unwrap()));
    }
}
