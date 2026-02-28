use std::collections::HashMap;
use std::path::Path;

use crate::error::SwarmError;

/// Current database schema version. Increment when making breaking changes
/// to the sled storage format.
pub const DB_SCHEMA_VERSION: u32 = 1;

/// Critical sled trees to check during integrity verification.
const CRITICAL_TREES: &[&str] = &["manifests", "credits", "identity", "nicknames"];

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

/// Wrapper around sled embedded database.
///
/// In Phase 1 this only stores persisted config. Later phases add
/// credit balances, peer trust, shard metadata, etc.
#[derive(Clone)]
pub struct Database {
    inner: sled::Db,
}

impl Database {
    /// Open (or create) the sled database at `data_dir/db`.
    pub fn open(data_dir: &Path) -> Result<Self, SwarmError> {
        let db_path = data_dir.join("db");
        let inner = sled::open(&db_path)?;
        tracing::info!(path = %db_path.display(), "Opened database");

        let db = Self { inner };
        db.check_schema_version()?;
        Ok(db)
    }

    /// Check and store the DB schema version. Warn on mismatch.
    fn check_schema_version(&self) -> Result<(), SwarmError> {
        let tree = self.tree("meta")?;
        match tree.get("schema_version")? {
            Some(bytes) => {
                if bytes.len() == 4 {
                    let stored = u32::from_le_bytes(bytes[..4].try_into().unwrap());
                    if stored != DB_SCHEMA_VERSION {
                        tracing::warn!(
                            stored_version = stored,
                            current_version = DB_SCHEMA_VERSION,
                            "Database schema version mismatch — data may need migration"
                        );
                    }
                }
            }
            None => {
                // First run — store the current version
                tree.insert("schema_version", &DB_SCHEMA_VERSION.to_le_bytes())?;
            }
        }
        Ok(())
    }

    /// Open a temporary in-memory database (for testing).
    pub fn open_temp() -> Result<Self, SwarmError> {
        let config = sled::Config::new().temporary(true);
        let inner = config.open()?;
        Ok(Self { inner })
    }

    /// Get a named tree (logical keyspace).
    pub fn tree(&self, name: &str) -> Result<sled::Tree, SwarmError> {
        Ok(self.inner.open_tree(name)?)
    }

    /// Store a JSON-serializable value.
    pub fn put_json<T: serde::Serialize>(
        &self,
        tree_name: &str,
        key: &str,
        value: &T,
    ) -> Result<(), SwarmError> {
        let tree = self.tree(tree_name)?;
        let bytes = serde_json::to_vec(value)?;
        tree.insert(key, bytes)?;
        Ok(())
    }

    /// Load a JSON-deserializable value.
    pub fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        tree_name: &str,
        key: &str,
    ) -> Result<Option<T>, SwarmError> {
        let tree = self.tree(tree_name)?;
        match tree.get(key)? {
            Some(bytes) => {
                let val = serde_json::from_slice(&bytes)?;
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
        let tree = self.tree(tree_name)?;
        let mut results = Vec::new();
        for entry in tree.iter() {
            let (key, bytes) = entry?;
            match serde_json::from_slice(&bytes) {
                Ok(val) => results.push(val),
                Err(e) => {
                    let key_str = std::str::from_utf8(&key)
                        .unwrap_or("<non-utf8>");
                    tracing::warn!(
                        tree = tree_name,
                        key = key_str,
                        error = %e,
                        "Failed to deserialize entry in iter_json, skipping"
                    );
                }
            }
        }
        Ok(results)
    }

    /// Remove a key from a named tree.
    pub fn remove(&self, tree_name: &str, key: &str) -> Result<(), SwarmError> {
        let tree = self.tree(tree_name)?;
        tree.remove(key)?;
        Ok(())
    }

    /// Flush all pending writes to disk.
    pub fn flush(&self) -> Result<(), SwarmError> {
        self.inner.flush()?;
        Ok(())
    }

    /// Check integrity of critical sled trees.
    ///
    /// Scans manifests, credits, identity, and nicknames trees.
    /// For each tree: opens it, iterates entries, attempts JSON deserialization.
    /// Logs warnings for any corrupt entries (includes key, skips value).
    pub fn check_integrity(&self) -> IntegrityReport {
        let mut trees = HashMap::new();
        let mut total_corrupt = 0;

        for &tree_name in CRITICAL_TREES {
            let tree = match self.tree(tree_name) {
                Ok(t) => t,
                Err(e) => {
                    tracing::warn!(tree = tree_name, error = %e, "Failed to open tree during integrity check");
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

            for entry in tree.iter() {
                match entry {
                    Ok((key, value)) => {
                        total += 1;
                        // Attempt JSON deserialization as generic Value
                        if serde_json::from_slice::<serde_json::Value>(&value).is_ok() {
                            valid += 1;
                        } else {
                            corrupt += 1;
                            let key_str =
                                std::str::from_utf8(&key).unwrap_or("<non-utf8>");
                            tracing::warn!(
                                tree = tree_name,
                                key = key_str,
                                "Corrupt entry detected during integrity check"
                            );
                        }
                    }
                    Err(e) => {
                        total += 1;
                        corrupt += 1;
                        tracing::warn!(
                            tree = tree_name,
                            error = %e,
                            "Failed to read entry during integrity check"
                        );
                    }
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
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

        db.put_json("test_tree", "key1", &serde_json::json!({"hello": "world"}))
            .unwrap();

        let val: Option<serde_json::Value> = db.get_json("test_tree", "key1").unwrap();
        assert_eq!(val.unwrap()["hello"], "world");
    }

    #[test]
    fn get_missing_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();

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
        // Insert raw invalid bytes directly into the tree
        let tree = db.tree("manifests").unwrap();
        tree.insert("corrupt_key", b"not valid json {{{" as &[u8])
            .unwrap();

        let report = db.check_integrity();
        assert_eq!(report.total_corrupt, 1);

        let manifests = report.trees.get("manifests").unwrap();
        assert_eq!(manifests.total_entries, 2);
        assert_eq!(manifests.valid_entries, 1);
        assert_eq!(manifests.corrupt_entries, 1);
    }
}
