use std::path::Path;

use crate::error::SwarmError;

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
        Ok(Self { inner })
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
            let (_, bytes) = entry?;
            if let Ok(val) = serde_json::from_slice(&bytes) {
                results.push(val);
            }
        }
        Ok(results)
    }

    /// Flush all pending writes to disk.
    pub fn flush(&self) -> Result<(), SwarmError> {
        self.inner.flush()?;
        Ok(())
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
}
