use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use redb::{ReadableDatabase, ReadableTable, TableDefinition};

use crate::error::SwarmError;

/// Critical trees to check during integrity verification.
const CRITICAL_TREES: &[&str] = &["manifests", "credits", "identity", "nicknames"];

/// Single redb table storing all logical trees via composite keys.
/// Key format: "{tree_name}\0{key}" (NUL separator).
const DATA_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("data");

/// Per-tree integrity status.
#[derive(Debug, Clone)]
pub struct TreeStatus {
    pub total_entries: usize,
    pub valid_entries: usize,
    pub corrupt_entries: usize,
}

/// Result of a full database integrity check.
#[derive(Debug, Clone)]
pub struct IntegrityReport {
    pub trees: HashMap<String, TreeStatus>,
    pub total_corrupt: usize,
}

/// Build a composite key: "{tree}\0{key}".
/// Panics if tree or key contain NUL bytes (would cause cross-tree collisions).
fn make_key(tree: &str, key: &str) -> Vec<u8> {
    // Defensive: strip NUL from tree name to prevent cross-tree key collisions.
    // Tree names are always compile-time string literals — this guard is a safety net.
    debug_assert!(
        !tree.contains('\0'),
        "DB tree name must not contain NUL bytes"
    );
    let mut k = Vec::with_capacity(tree.len() + 1 + key.len());
    k.extend_from_slice(tree.as_bytes());
    k.push(0);
    // Strip NUL bytes from key defensively (gossip-sourced model IDs)
    k.extend(key.as_bytes().iter().filter(|&&b| b != 0));
    k
}

/// Build the start of a tree prefix range (inclusive): "{tree}\0".
fn tree_range_start(tree: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(tree.len() + 1);
    k.extend_from_slice(tree.as_bytes());
    k.push(0);
    k
}

/// Build the end of a tree prefix range (exclusive): "{tree}\x01".
fn tree_range_end(tree: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(tree.len() + 1);
    k.extend_from_slice(tree.as_bytes());
    k.push(1);
    k
}

/// Extract the sub-key from a composite key (everything after the first NUL byte).
fn extract_subkey(composite: &[u8]) -> Option<&[u8]> {
    composite
        .iter()
        .position(|&b| b == 0)
        .map(|pos| &composite[pos + 1..])
}

/// Wrapper around redb embedded database.
///
/// Uses a single redb table with composite keys ("{tree}\0{key}") to emulate
/// a named-tree pattern. All values are stored as raw bytes (typically JSON).
#[derive(Clone)]
pub struct Database {
    inner: Arc<redb::Database>,
    /// Holds the temp directory (or temp file guard) alive for `open_temp()`
    /// databases. `Arc` lets the guard survive across Database clones;
    /// the last drop releases the underlying file.
    _temp_dir: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

/// RAII guard for `open_temp()` files — removes the path on drop so test
/// runs don't leak redb files into the system temp directory.
struct TempFileGuard(std::path::PathBuf);
impl Drop for TempFileGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

impl Database {
    /// Open (or create) the redb database at `data_dir/db.redb`.
    ///
    /// If the file uses an old format, it is deleted and recreated.
    pub fn open(data_dir: &Path) -> Result<Self, SwarmError> {
        let db_path = data_dir.join("db.redb");
        let inner = match redb::Database::create(&db_path) {
            Ok(db) => db,
            Err(redb::DatabaseError::UpgradeRequired(version)) => {
                let backup_path = db_path.with_extension("redb.bak");
                tracing::warn!(
                    old_version = version,
                    backup = %backup_path.display(),
                    "Old redb v{version} database found; backing up and recreating"
                );
                if let Err(e) = std::fs::rename(&db_path, &backup_path) {
                    tracing::error!(error = %e, "Failed to backup old database — deleting");
                    let _ = std::fs::remove_file(&db_path);
                }
                redb::Database::create(&db_path).map_err(|e| {
                    SwarmError::Database(format!("Failed to create {}: {e}", db_path.display()))
                })?
            }
            Err(e) => {
                return Err(SwarmError::Database(format!(
                    "Failed to open {}: {e}",
                    db_path.display()
                )));
            }
        };
        // SEC: enforce 0o600 on Unix. The redb file holds plaintext credit
        // balances, peer trust scores, and manifest metadata (provider keys
        // are encrypted, but the rest is sensitive). Process umask defaults
        // to 0o022 on most distros, leaving the file world-readable.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(e) =
                std::fs::set_permissions(&db_path, std::fs::Permissions::from_mode(0o600))
            {
                tracing::warn!(path = %db_path.display(), error = %e, "Failed to chmod db.redb to 0600");
            }
        }
        tracing::debug!(path = %db_path.display(), "DIAG: db_open");

        let db = Self {
            inner: Arc::new(inner),
            _temp_dir: None,
        };
        Ok(db)
    }

    /// Open a temporary database (for testing).
    ///
    /// Creates a uniquely-named database in the system temp directory. The
    /// path is held alive by `_temp_dir` via a small RAII guard that removes
    /// the file when the last `Database` clone drops, so test runs don't
    /// leak redb files into `/tmp`.
    pub fn open_temp() -> Result<Self, SwarmError> {
        let temp_path =
            std::env::temp_dir().join(format!("swarmllm_test_{}.redb", uuid::Uuid::new_v4()));
        let inner = redb::Database::create(&temp_path)
            .map_err(|e| SwarmError::Database(format!("Failed to create temp db: {e}")))?;
        Ok(Self {
            inner: Arc::new(inner),
            _temp_dir: Some(Arc::new(TempFileGuard(temp_path))),
        })
    }

    /// Open a write transaction, run `f` on the data table, commit on
    /// `Ok`, rollback (drop) on `Err`. Removes the open-write/open-table/
    /// commit boilerplate from every mutating method (R96).
    fn with_write_table<R>(
        &self,
        f: impl FnOnce(&mut redb::Table<'_, &'static [u8], &'static [u8]>) -> Result<R, SwarmError>,
    ) -> Result<R, SwarmError> {
        let txn = self
            .inner
            .begin_write()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let result = {
            let mut table = txn
                .open_table(DATA_TABLE)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            f(&mut table)?
        };
        txn.commit()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        Ok(result)
    }

    /// Store a JSON-serializable value.
    pub fn put_json<T: serde::Serialize>(
        &self,
        tree_name: &str,
        key: &str,
        value: &T,
    ) -> Result<(), SwarmError> {
        let k = make_key(tree_name, key);
        let bytes = serde_json::to_vec(value)?;
        self.with_write_table(|table| {
            table
                .insert(k.as_slice(), bytes.as_slice())
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            Ok(())
        })
    }

    /// Load a JSON-deserializable value.
    pub fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        tree_name: &str,
        key: &str,
    ) -> Result<Option<T>, SwarmError> {
        let k = make_key(tree_name, key);
        let read_txn = self
            .inner
            .begin_read()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let table = match read_txn.open_table(DATA_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => return Err(SwarmError::Database(e.to_string())),
        };
        match table
            .get(k.as_slice())
            .map_err(|e| SwarmError::Database(e.to_string()))?
        {
            Some(guard) => {
                let val = serde_json::from_slice(guard.value())?;
                Ok(Some(val))
            }
            None => Ok(None),
        }
    }

    /// Iterate all values in a named tree, deserializing each as JSON.
    pub fn iter_json<T: serde::de::DeserializeOwned>(
        &self,
        tree_name: &str,
    ) -> Result<Vec<T>, SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        let read_txn = self
            .inner
            .begin_read()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let table = match read_txn.open_table(DATA_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(SwarmError::Database(e.to_string())),
        };

        let mut results = Vec::new();
        let range = table
            .range(start.as_slice()..end.as_slice())
            .map_err(|e| SwarmError::Database(e.to_string()))?;

        for entry in range {
            let (key_guard, val_guard) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
            match serde_json::from_slice(val_guard.value()) {
                Ok(val) => results.push(val),
                Err(e) => {
                    let subkey = extract_subkey(key_guard.value())
                        .and_then(|b| std::str::from_utf8(b).ok())
                        .unwrap_or("<non-utf8>");
                    tracing::warn!(
                        tree = tree_name,
                        key = subkey,
                        error = %e,
                        "Failed to deserialize entry in iter_json, skipping"
                    );
                }
            }
        }
        Ok(results)
    }

    /// Iterate all key-value pairs in a named tree, returning (subkey_string, deserialized_value).
    pub fn get_all_json<T: serde::de::DeserializeOwned>(
        &self,
        tree_name: &str,
    ) -> Result<Vec<(String, T)>, SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        let read_txn = self
            .inner
            .begin_read()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let table = match read_txn.open_table(DATA_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(SwarmError::Database(e.to_string())),
        };

        let mut results = Vec::new();
        let range = table
            .range(start.as_slice()..end.as_slice())
            .map_err(|e| SwarmError::Database(e.to_string()))?;

        for entry in range {
            let (key_guard, val_guard) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
            let subkey = extract_subkey(key_guard.value())
                .and_then(|b| std::str::from_utf8(b).ok())
                .unwrap_or("")
                .to_string();
            match serde_json::from_slice(val_guard.value()) {
                Ok(val) => results.push((subkey, val)),
                Err(e) => {
                    tracing::warn!(
                        tree = tree_name,
                        key = %subkey,
                        error = %e,
                        "Failed to deserialize entry in get_all_json, skipping"
                    );
                }
            }
        }
        Ok(results)
    }

    /// Stream every JSON-encoded record in a tree through `f` without
    /// materialising a full `Vec`. Lets the caller maintain a bounded
    /// data structure (heap, top-k cache, running aggregate) so listings
    /// over large trees stay O(N) in time but O(k) in memory.
    /// Skipping is mid-flight: deserialization failures emit a warning
    /// and continue, matching `iter_json` / `get_all_json` semantics.
    pub fn for_each_json<T, F>(&self, tree_name: &str, mut f: F) -> Result<(), SwarmError>
    where
        T: serde::de::DeserializeOwned,
        F: FnMut(&str, T),
    {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        let read_txn = self
            .inner
            .begin_read()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let table = match read_txn.open_table(DATA_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(SwarmError::Database(e.to_string())),
        };
        let range = table
            .range(start.as_slice()..end.as_slice())
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        for entry in range {
            let (key_guard, val_guard) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
            let subkey = extract_subkey(key_guard.value())
                .and_then(|b| std::str::from_utf8(b).ok())
                .unwrap_or("");
            match serde_json::from_slice::<T>(val_guard.value()) {
                Ok(val) => f(subkey, val),
                Err(e) => {
                    tracing::warn!(
                        tree = tree_name,
                        key = %subkey,
                        error = %e,
                        "Failed to deserialize entry in for_each_json, skipping"
                    );
                }
            }
        }
        Ok(())
    }

    /// Iterate all raw key-value pairs in a named tree.
    /// Returns (sub_key_bytes, value_bytes) pairs.
    #[allow(clippy::type_complexity)]
    pub fn iter_raw(&self, tree_name: &str) -> Result<Vec<(Vec<u8>, Vec<u8>)>, SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        let read_txn = self
            .inner
            .begin_read()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        let table = match read_txn.open_table(DATA_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => return Err(SwarmError::Database(e.to_string())),
        };

        let mut results = Vec::new();
        let range = table
            .range(start.as_slice()..end.as_slice())
            .map_err(|e| SwarmError::Database(e.to_string()))?;

        for entry in range {
            let (key_guard, val_guard) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
            let subkey = extract_subkey(key_guard.value())
                .unwrap_or_default()
                .to_vec();
            results.push((subkey, val_guard.value().to_vec()));
        }
        Ok(results)
    }

    /// Insert a raw byte value into a named tree.
    pub fn insert_raw(&self, tree_name: &str, key: &str, value: &[u8]) -> Result<(), SwarmError> {
        let k = make_key(tree_name, key);
        self.with_write_table(|table| {
            table
                .insert(k.as_slice(), value)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            Ok(())
        })
    }

    /// Remove a key from a named tree.
    pub fn remove(&self, tree_name: &str, key: &str) -> Result<(), SwarmError> {
        let k = make_key(tree_name, key);
        self.with_write_table(|table| {
            table
                .remove(k.as_slice())
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            Ok(())
        })
    }

    /// Clear all entries from a named tree.
    pub fn clear_tree(&self, tree_name: &str) -> Result<(), SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        self.with_write_table(|table| {
            // Collect keys to remove (can't mutate while iterating)
            let keys: Vec<Vec<u8>> = {
                let range = table
                    .range(start.as_slice()..end.as_slice())
                    .map_err(|e| SwarmError::Database(e.to_string()))?;
                let mut ks = Vec::new();
                for entry in range {
                    let (key_guard, _) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
                    ks.push(key_guard.value().to_vec());
                }
                ks
            };
            for key in &keys {
                table
                    .remove(key.as_slice())
                    .map_err(|e| SwarmError::Database(e.to_string()))?;
            }
            Ok(())
        })
    }

    /// Atomically replace every entry under `tree_name` with the given
    /// key/value pairs in a single redb write transaction. Either the
    /// whole new set lands or none of it does — readers never observe a
    /// partially-cleared tree.
    ///
    /// `peer_cache` and any other "snapshot list" persistence path
    /// should use this in preference to `clear_tree` + N `insert_raw`
    /// calls (which span N+1 transactions and can leave the tree empty
    /// or partially populated if the process is killed mid-write).
    pub fn replace_tree(
        &self,
        tree_name: &str,
        entries: &[(String, Vec<u8>)],
    ) -> Result<(), SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        self.with_write_table(|table| {
            // Collect existing keys then remove them — same pattern as
            // clear_tree, but inside the same txn as the inserts.
            let stale_keys: Vec<Vec<u8>> = {
                let range = table
                    .range(start.as_slice()..end.as_slice())
                    .map_err(|e| SwarmError::Database(e.to_string()))?;
                let mut ks = Vec::new();
                for entry in range {
                    let (key_guard, _) = entry.map_err(|e| SwarmError::Database(e.to_string()))?;
                    ks.push(key_guard.value().to_vec());
                }
                ks
            };
            for key in &stale_keys {
                table
                    .remove(key.as_slice())
                    .map_err(|e| SwarmError::Database(e.to_string()))?;
            }

            for (subkey, value) in entries {
                let k = make_key(tree_name, subkey);
                table
                    .insert(k.as_slice(), value.as_slice())
                    .map_err(|e| SwarmError::Database(e.to_string()))?;
            }
            Ok(())
        })
    }

    /// Persist the user's --shards range so it's restored on next startup.
    pub fn save_shard_range(&self, start: u32, end: u32) -> Result<(), SwarmError> {
        self.put_json("config", "shard_range", &(start, end))
    }

    /// Load a previously persisted shard range, if any.
    pub fn load_shard_range(&self) -> Result<Option<(u32, u32)>, SwarmError> {
        self.get_json("config", "shard_range")
    }

    /// Check integrity of critical trees.
    ///
    /// Scans manifests, credits, identity, and nicknames trees.
    /// For each tree: iterates entries, attempts JSON deserialization.
    /// Logs warnings for any corrupt entries.
    pub fn check_integrity(&self) -> IntegrityReport {
        let mut trees = HashMap::new();
        let mut total_corrupt = 0;

        for &tree_name in CRITICAL_TREES {
            let entries = match self.iter_raw(tree_name) {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!(tree = tree_name, error = %e, "Failed to read tree during integrity check");
                    trees.insert(
                        tree_name.to_string(),
                        TreeStatus {
                            total_entries: 0,
                            valid_entries: 0,
                            corrupt_entries: 0,
                        },
                    );
                    continue;
                }
            };

            let mut total = 0;
            let mut valid = 0;
            let mut corrupt = 0;

            for (key_bytes, value) in &entries {
                total += 1;
                if serde_json::from_slice::<serde_json::Value>(value).is_ok() {
                    valid += 1;
                } else {
                    corrupt += 1;
                    let key_str = std::str::from_utf8(key_bytes).unwrap_or("<non-utf8>");
                    tracing::warn!(
                        tree = tree_name,
                        key = key_str,
                        "Corrupt entry detected during integrity check"
                    );
                }
            }

            total_corrupt += corrupt;
            trees.insert(
                tree_name.to_string(),
                TreeStatus {
                    total_entries: total,
                    valid_entries: valid,
                    corrupt_entries: corrupt,
                },
            );
        }

        if total_corrupt > 0 {
            tracing::warn!(
                total_corrupt,
                "Database integrity check found corrupt entries"
            );
        } else {
            tracing::info!("Database integrity check passed");
        }

        IntegrityReport {
            trees,
            total_corrupt,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get_json() {
        let db = Database::open_temp().unwrap();

        db.put_json("test_tree", "key1", &serde_json::json!({"hello": "world"}))
            .unwrap();

        let val: Option<serde_json::Value> = db.get_json("test_tree", "key1").unwrap();
        assert_eq!(val.unwrap()["hello"], "world");
    }

    #[test]
    fn get_missing_key_returns_none() {
        let db = Database::open_temp().unwrap();

        let val: Option<serde_json::Value> = db.get_json("test_tree", "missing").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn integrity_check_empty_trees() {
        let db = Database::open_temp().unwrap();
        let report = db.check_integrity();
        assert_eq!(report.total_corrupt, 0);
        for tree_name in CRITICAL_TREES {
            let status = report.trees.get(*tree_name).unwrap();
            assert_eq!(status.total_entries, 0);
            assert_eq!(status.valid_entries, 0);
            assert_eq!(status.corrupt_entries, 0);
        }
    }

    #[test]
    fn integrity_check_valid_entries() {
        let db = Database::open_temp().unwrap();
        db.put_json("manifests", "model1", &serde_json::json!({"name": "test"}))
            .unwrap();
        db.put_json("credits", "node1", &serde_json::json!({"balance": 100}))
            .unwrap();

        let report = db.check_integrity();
        assert_eq!(report.total_corrupt, 0);

        let manifests = report.trees.get("manifests").unwrap();
        assert_eq!(manifests.total_entries, 1);
        assert_eq!(manifests.valid_entries, 1);
        assert_eq!(manifests.corrupt_entries, 0);
    }

    #[test]
    fn integrity_check_detects_corrupt_entry() {
        let db = Database::open_temp().unwrap();
        // Insert a valid JSON entry
        db.put_json("manifests", "good", &serde_json::json!({"ok": true}))
            .unwrap();
        // Insert raw invalid bytes directly
        db.insert_raw("manifests", "corrupt_key", b"not valid json {{{")
            .unwrap();

        let report = db.check_integrity();
        assert_eq!(report.total_corrupt, 1);

        let manifests = report.trees.get("manifests").unwrap();
        assert_eq!(manifests.total_entries, 2);
        assert_eq!(manifests.valid_entries, 1);
        assert_eq!(manifests.corrupt_entries, 1);
    }

    #[test]
    fn save_and_load_shard_range() {
        let db = Database::open_temp().unwrap();
        db.save_shard_range(0, 4).unwrap();
        let loaded = db.load_shard_range().unwrap();
        assert_eq!(loaded, Some((0, 4)));
    }

    #[test]
    fn load_shard_range_returns_none_when_not_set() {
        let db = Database::open_temp().unwrap();
        let loaded = db.load_shard_range().unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn save_shard_range_overwrites_previous() {
        let db = Database::open_temp().unwrap();
        db.save_shard_range(0, 4).unwrap();
        db.save_shard_range(5, 8).unwrap();
        let loaded = db.load_shard_range().unwrap();
        assert_eq!(loaded, Some((5, 8)));
    }

    #[test]
    fn replace_tree_swaps_contents_atomically() {
        let db = Database::open_temp().unwrap();
        let tree = "snapshot";

        // Seed initial set.
        let initial = vec![
            ("a".to_string(), b"v1".to_vec()),
            ("b".to_string(), b"v2".to_vec()),
        ];
        db.replace_tree(tree, &initial).unwrap();
        assert_eq!(db.iter_raw(tree).unwrap().len(), 2);

        // Replace with a fresh set; old keys should not survive.
        let next = vec![("c".to_string(), b"v3".to_vec())];
        db.replace_tree(tree, &next).unwrap();
        let entries = db.iter_raw(tree).unwrap();
        assert_eq!(entries.len(), 1);
        let (_, v) = &entries[0];
        assert_eq!(v.as_slice(), b"v3");
    }

    #[test]
    fn replace_tree_with_empty_clears() {
        let db = Database::open_temp().unwrap();
        let tree = "snapshot";
        db.replace_tree(
            tree,
            &[
                ("a".to_string(), b"v1".to_vec()),
                ("b".to_string(), b"v2".to_vec()),
            ],
        )
        .unwrap();
        assert_eq!(db.iter_raw(tree).unwrap().len(), 2);

        db.replace_tree(tree, &[]).unwrap();
        assert!(db.iter_raw(tree).unwrap().is_empty());
    }

    #[test]
    fn iter_raw_roundtrip() {
        let db = Database::open_temp().unwrap();
        db.insert_raw("raw_tree", "key1", b"value1").unwrap();
        db.insert_raw("raw_tree", "key2", b"value2").unwrap();

        let entries = db.iter_raw("raw_tree").unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn clear_tree_removes_all() {
        let db = Database::open_temp().unwrap();
        db.put_json("to_clear", "a", &1).unwrap();
        db.put_json("to_clear", "b", &2).unwrap();
        db.put_json("other", "c", &3).unwrap();

        db.clear_tree("to_clear").unwrap();

        let cleared: Vec<serde_json::Value> = db.iter_json("to_clear").unwrap();
        assert!(cleared.is_empty());

        // Other tree should be unaffected
        let other: Option<serde_json::Value> = db.get_json("other", "c").unwrap();
        assert!(other.is_some());
    }

    #[test]
    fn open_on_disk() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        db.put_json("test", "k", &42u32).unwrap();
        drop(db);

        // Reopen and verify data persists
        let db2 = Database::open(dir.path()).unwrap();
        let val: Option<u32> = db2.get_json("test", "k").unwrap();
        assert_eq!(val, Some(42));
    }
}
