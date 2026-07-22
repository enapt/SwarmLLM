use libp2p::multiaddr::Protocol;
use libp2p::{Multiaddr, PeerId};

use crate::network::manager::addr_is_remotely_reachable;
use crate::storage::db::Database;

/// Database tree name for cached peer addresses.
const TREE_PEER_CACHE: &str = "peer_cache";

/// Maximum number of peers to persist in the cache.
const MAX_CACHED_PEERS: usize = 200;

/// Drop cached addresses that can never produce a useful dial.
///
/// The cache is built from peers' identify-advertised listen addresses, which
/// include whatever private interfaces each peer happens to have. Storing them
/// verbatim meant a public node re-dialled a *remote* peer's `127.0.0.1`,
/// docker-bridge and libvirt addresses on every retry, indefinitely — observed
/// on the live anchor, which also never recovered because `save_peer_cache`
/// skips the write while `peer_registry` is empty, so the bad entries were
/// never overwritten.
///
/// Two classes are removed:
/// - Not remotely reachable at all — loopback, unspecified, link-local, cloud
///   metadata. Shares [`addr_is_remotely_reachable`] with the advertise path.
/// - Anything naming *us*: a `/p2p/<local>` target, or a relay circuit whose
///   relay hop is our own peer id. Both are dials to self; the relay form
///   showed up as the anchor trying to reach a peer by relaying through
///   itself.
///
/// LAN addresses are deliberately KEPT. They are useless on a VPS but they are
/// how two machines in one house find each other again after a reboot, and
/// this cache serves both deployments.
pub fn filter_dialable(addrs: &[String], local_peer_id: &PeerId) -> Vec<String> {
    addrs
        .iter()
        .filter(|s| {
            s.parse::<Multiaddr>()
                .map(|a| is_dialable(&a, local_peer_id))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}

fn is_dialable(addr: &Multiaddr, local_peer_id: &PeerId) -> bool {
    // An empty string parses into a valid Multiaddr carrying no protocols, and
    // every predicate below vacuously passes it. It names no destination.
    if addr.iter().next().is_none() {
        return false;
    }
    if !addr_is_remotely_reachable(addr) {
        return false;
    }
    // Every `/p2p/` hop, not just the final one — the relay position in a
    // `/p2p-circuit` address is what made the anchor route through itself.
    !addr
        .iter()
        .any(|p| matches!(p, Protocol::P2p(pid) if pid == *local_peer_id))
}

/// Save known peer multiaddrs to the database for reconnection on restart.
///
/// Stores up to `MAX_CACHED_PEERS` addresses keyed by sequential index.
/// Uses `replace_tree` so the clear + N inserts land atomically — a
/// crash mid-save never leaves the cache empty or partially populated.
pub fn save_peer_cache(db: &Database, addrs: &[String]) {
    let entries: Vec<(String, Vec<u8>)> = addrs
        .iter()
        .take(MAX_CACHED_PEERS)
        .enumerate()
        .map(|(i, addr)| (format!("peer_{i:04}"), addr.as_bytes().to_vec()))
        .collect();

    let count = entries.len();
    if let Err(e) = db.replace_tree(TREE_PEER_CACHE, &entries) {
        tracing::warn!(error = %e, "Failed to persist peer cache");
        return;
    }

    tracing::debug!(count, "DIAG: peer cache saved");
}

/// Load cached peer multiaddrs from the database.
///
/// Returns addresses from the last session for immediate reconnection.
pub fn load_peer_cache(db: &Database) -> Vec<String> {
    let entries = match db.iter_raw(TREE_PEER_CACHE) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read peer cache");
            return Vec::new();
        }
    };

    let mut addrs = Vec::new();
    for (_key, value) in entries {
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

    fn pid(seed: u8) -> PeerId {
        libp2p::identity::Keypair::ed25519_from_bytes([seed; 32])
            .unwrap()
            .public()
            .to_peer_id()
    }

    // ── filter_dialable ──
    //
    // The addresses below are the real ones observed being re-dialled every
    // 60s by the public anchor node (peer 12D3KooWNisn…), which had cached a
    // remote peer's loopback + docker + libvirt addresses and a relay circuit
    // through its own peer id.

    #[test]
    fn drops_loopback_and_keeps_public() {
        let me = pid(1);
        let other = pid(2);
        let addrs = vec![
            format!("/ip4/127.0.0.1/tcp/8810/p2p/{other}"),
            format!("/ip4/81.241.51.1/tcp/8810/p2p/{other}"),
        ];
        let kept = filter_dialable(&addrs, &me);
        assert_eq!(kept, vec![format!("/ip4/81.241.51.1/tcp/8810/p2p/{other}")]);
    }

    #[test]
    fn keeps_lan_and_cgn_addresses() {
        // Useless on a VPS, but this is how two machines in one house find
        // each other after a reboot. Deliberately retained.
        let me = pid(1);
        let other = pid(2);
        let addrs = vec![
            format!("/ip4/192.168.129.3/tcp/8810/p2p/{other}"),
            format!("/ip4/172.17.0.1/tcp/8810/p2p/{other}"),
            format!("/ip4/100.116.22.41/udp/8800/quic-v1/p2p/{other}"),
        ];
        assert_eq!(filter_dialable(&addrs, &me).len(), 3);
    }

    #[test]
    fn drops_addresses_targeting_ourselves() {
        let me = pid(1);
        let addrs = vec![format!("/ip4/212.132.104.177/tcp/8810/p2p/{me}")];
        assert!(filter_dialable(&addrs, &me).is_empty());
    }

    #[test]
    fn drops_relay_circuits_through_ourselves() {
        // The relay hop is not the final /p2p/ component, so a check that only
        // inspected the target peer id would let this through.
        let me = pid(1);
        let other = pid(2);
        let addrs = vec![format!(
            "/dns4/swarmllm.duckdns.org/tcp/8810/p2p/{me}/p2p-circuit/p2p/{other}"
        )];
        assert!(filter_dialable(&addrs, &me).is_empty());
    }

    #[test]
    fn keeps_relay_circuits_through_someone_else() {
        let me = pid(1);
        let relay = pid(2);
        let target = pid(3);
        let addrs = vec![format!(
            "/dns4/relay.example.net/tcp/8810/p2p/{relay}/p2p-circuit/p2p/{target}"
        )];
        assert_eq!(filter_dialable(&addrs, &me).len(), 1);
    }

    #[test]
    fn drops_unparseable_addresses() {
        let me = pid(1);
        let addrs = vec!["not-a-multiaddr".to_string(), String::new()];
        assert!(filter_dialable(&addrs, &me).is_empty());
    }

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
