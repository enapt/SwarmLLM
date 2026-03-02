use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use redb::{ReadableTable, TableDefinition};

use crate::error::SwarmError;

/// Current database schema version. Increment when making breaking changes.
/// Version 2: migrated from sled to redb.
pub const DB_SCHEMA_VERSION: u32 = 2;

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
fn make_key(tree: &str, key: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(tree.len() + 1 + key.len());
    k.extend_from_slice(tree.as_bytes());
    k.push(0);
    k.extend_from_slice(key.as_bytes());
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
/// sled's named trees. All values are stored as raw bytes (typically JSON).
#[derive(Clone)]
pub struct Database {
    inner: Arc<redb::Database>,
    /// Holds the temp directory alive for `open_temp()` databases.
    _temp_dir: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl Database {
    /// Open (or create) the redb database at `data_dir/db.redb`.
    pub fn open(data_dir: &Path) -> Result<Self, SwarmError> {
        let db_path = data_dir.join("db.redb");
        let inner = redb::Database::create(&db_path).map_err(|e| {
            SwarmError::Database(format!("Failed to open {}: {e}", db_path.display()))
        })?;
        tracing::info!(path = %db_path.display(), "Opened database");

        let db = Self {
            inner: Arc::new(inner),
            _temp_dir: None,
        };
        db.check_schema_version()?;
        Ok(db)
    }

    /// Check and store the DB schema version. Warn on mismatch.
    fn check_schema_version(&self) -> Result<(), SwarmError> {
        match self.get_json::<u32>("meta", "schema_version")? {
            Some(stored) => {
                if stored != DB_SCHEMA_VERSION {
                    tracing::warn!(
                        stored_version = stored,
                        current_version = DB_SCHEMA_VERSION,
                        "Database schema version mismatch — data may need migration"
                    );
                }
            }
            None => {
                // First run — store the current version
                self.put_json("meta", "schema_version", &DB_SCHEMA_VERSION)?;
            }
        }
        Ok(())
    }

    /// Open a temporary database (for testing).
    ///
    /// Creates a uniquely-named database in the system temp directory.
    /// The file is not auto-deleted; callers should use `tempfile::tempdir()`
    /// + `Database::open()` if cleanup is needed.
    pub fn open_temp() -> Result<Self, SwarmError> {
        let temp_path =
            std::env::temp_dir().join(format!("swarmllm_test_{}.redb", uuid::Uuid::new_v4()));
        let inner = redb::Database::create(&temp_path)
            .map_err(|e| SwarmError::Database(format!("Failed to create temp db: {e}")))?;
        Ok(Self {
            inner: Arc::new(inner),
            _temp_dir: None,
        })
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
        let write_txn = self
            .inner
            .begin_write()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DATA_TABLE)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            table
                .insert(k.as_slice(), bytes.as_slice())
                .map_err(|e| SwarmError::Database(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        Ok(())
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
        let write_txn = self
            .inner
            .begin_write()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DATA_TABLE)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            table
                .insert(k.as_slice(), value)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        Ok(())
    }

    /// Remove a key from a named tree.
    pub fn remove(&self, tree_name: &str, key: &str) -> Result<(), SwarmError> {
        let k = make_key(tree_name, key);
        let write_txn = self
            .inner
            .begin_write()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DATA_TABLE)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
            table
                .remove(k.as_slice())
                .map_err(|e| SwarmError::Database(e.to_string()))?;
        }
        write_txn
            .commit()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        Ok(())
    }

    /// Flush all pending writes to disk.
    /// With redb, writes are durable on commit, so this is a no-op.
    pub fn flush(&self) -> Result<(), SwarmError> {
        Ok(())
    }

    /// Clear all entries from a named tree.
    pub fn clear_tree(&self, tree_name: &str) -> Result<(), SwarmError> {
        let start = tree_range_start(tree_name);
        let end = tree_range_end(tree_name);
        let write_txn = self
            .inner
            .begin_write()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        {
            let mut table = write_txn
                .open_table(DATA_TABLE)
                .map_err(|e| SwarmError::Database(e.to_string()))?;
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
        }
        write_txn
            .commit()
            .map_err(|e| SwarmError::Database(e.to_string()))?;
        Ok(())
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
