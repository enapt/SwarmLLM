use crate::storage::db::Database;

/// Sled tree name for cached peer addresses.
const TREE_PEER_CACHE: &str = "peer_cache";

/// Maximum number of peers to persist in the cache.
const MAX_CACHED_PEERS: usize = 200;

/// Save known peer multiaddrs to sled for reconnection on restart.
///
/// Stores up to `MAX_CACHED_PEERS` addresses keyed by multiaddr string.
pub fn save_peer_cache(db: &Database, addrs: &[String]) {
    let tree = match db.tree(TREE_PEER_CACHE) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to open peer cache tree");
            return;
        }
    };

    // Clear old entries and write the current set
    tree.clear().ok();

    for (i, addr) in addrs.iter().take(MAX_CACHED_PEERS).enumerate() {
        let key = format!("peer_{i:04}");
        if let Err(e) = tree.insert(key.as_bytes(), addr.as_bytes()) {
            tracing::warn!(error = %e, addr, "Failed to cache peer address");
        }
    }

    if let Err(e) = tree.flush() {
        tracing::warn!(error = %e, "Failed to flush peer cache");
    }

    tracing::debug!(
        count = addrs.len().min(MAX_CACHED_PEERS),
        "Saved peer cache"
    );
}

/// Load cached peer multiaddrs from sled.
///
/// Returns addresses from the last session for immediate reconnection.
pub fn load_peer_cache(db: &Database) -> Vec<String> {
    let tree = match db.tree(TREE_PEER_CACHE) {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to open peer cache tree");
            return Vec::new();
        }
    };

    let mut addrs = Vec::new();
    for entry in tree.iter().flatten() {
        let (_key, value) = entry;
        if let Ok(addr) = std::str::from_utf8(&value) {
            addrs.push(addr.to_string());
        }
    }

    if !addrs.is_empty() {
        tracing::info!(count = addrs.len(), "Loaded cached peers from last session");
    }

    addrs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_roundtrip() {
        let db = Database::open_temp().unwrap();
        let addrs = vec![
            "/ip4/192.168.1.1/udp/8800/quic-v1".to_string(),
            "/ip4/10.0.0.5/udp/8800/quic-v1".to_string(),
        ];

        save_peer_cache(&db, &addrs);
        let loaded = load_peer_cache(&db);

        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains(&addrs[0]));
        assert!(loaded.contains(&addrs[1]));
    }

    #[test]
    fn empty_cache_returns_empty() {
        let db = Database::open_temp().unwrap();
        let loaded = load_peer_cache(&db);
        assert!(loaded.is_empty());
    }

    #[test]
    fn save_overwrites_previous() {
        let db = Database::open_temp().unwrap();

        save_peer_cache(&db, &["/ip4/1.1.1.1/udp/8800/quic-v1".to_string()]);
        save_peer_cache(&db, &["/ip4/2.2.2.2/udp/8800/quic-v1".to_string()]);

        let loaded = load_peer_cache(&db);
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0], "/ip4/2.2.2.2/udp/8800/quic-v1");
    }

    #[test]
    fn respects_max_cached_peers() {
        let db = Database::open_temp().unwrap();
        let addrs: Vec<String> = (0..300)
            .map(|i| format!("/ip4/10.0.{}.{}/udp/8800/quic-v1", i / 256, i % 256))
            .collect();

        save_peer_cache(&db, &addrs);
        let loaded = load_peer_cache(&db);

        assert_eq!(loaded.len(), MAX_CACHED_PEERS);
    }
}
