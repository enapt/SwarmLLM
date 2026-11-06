//! DHT + PEX handlers — sibling of manager/mod.rs.
//!
//! `handle_pex_response` dials peers from the inbound exchanged-peers list.
//! `handle_dht_provider_query` issues GetProviders for every shard of a
//! model. `handle_dht_providers_found` merges resolved providers into the
//! model_registry's bounded shard_holders cache.
//!
//! State (`pending_provider_queries`, `swarm`, `peer_to_node_id`) lives on
//! `NetworkManager` in mod.rs; this file is a `pub(super) impl` block.

use libp2p::{Multiaddr, PeerId};

use crate::network::helpers::{is_non_public_addr, target_peer_from_address};

use super::{NetworkManager, MAX_PENDING_PROVIDER_QUERIES};

/// The (peer, address) pair a failed dial proved wrong: the handshake at
/// `address` answered as ANOTHER computer. Only `WrongPeerId` proves it — a
/// timeout or a refused connection says nothing about who lives there now.
pub(super) fn pair_proven_wrong(
    expected: Option<PeerId>,
    error: &libp2p::swarm::DialError,
) -> Option<(PeerId, &Multiaddr)> {
    match (expected, error) {
        (Some(expected), libp2p::swarm::DialError::WrongPeerId { obtained, address })
            if *obtained != expected =>
        {
            Some((expected, address))
        }
        _ => None,
    }
}

/// Whether `held` (a registry address, with or without its `/p2p/` part) is
/// `address` for `peer` — compared as Kademlia compares them, `/p2p/` added.
fn is_this_address_of(held: &str, peer: PeerId, address: &Multiaddr) -> bool {
    let Ok(held) = held.parse::<Multiaddr>() else {
        return false;
    };
    matches!(
        (held.with_p2p(peer), address.clone().with_p2p(peer)),
        (Ok(a), Ok(b)) if a == b
    )
}

/// Forget `address` for `peer` in this node's peer registry — what the peer
/// cache saves and what peer exchange hands out. Returns how many went.
fn forget_in_registry(
    registry: &dashmap::DashMap<crate::types::NodeId, crate::types::PeerInfo>,
    peer: PeerId,
    address: &Multiaddr,
) -> usize {
    let peer_bytes = peer.to_bytes();
    // Found under read locks, then each entry written alone: `iter_mut` would
    // write-lock every shard of the registry in turn, from the network loop.
    let entries: Vec<crate::types::NodeId> = registry
        .iter()
        .filter(|entry| entry.peer_id_bytes.as_deref() == Some(peer_bytes.as_slice()))
        .map(|entry| entry.key().clone())
        .collect();
    let mut forgot = 0;
    for node in entries {
        if let Some(mut entry) = registry.get_mut(&node) {
            let before = entry.addresses.len();
            entry
                .addresses
                .retain(|held| !is_this_address_of(held, peer, address));
            forgot += before - entry.addresses.len();
        }
    }
    forgot
}

impl NetworkManager {
    /// A dial to `peer` at `address` reached ANOTHER computer: forget the pair
    /// everywhere this node keeps it, so nothing dials it, saves it or passes it
    /// on again.
    ///
    /// libp2p-kad 0.48 forgets an address only after a transport error
    /// (`on_dial_failure` → `address_failed`, for `DialError::Transport` alone),
    /// and never a peer's last one. So an identity that once listened where
    /// another computer now does stays in the routing table, is dialled by every
    /// lookup that picks it, and is handed to every node that asks. On
    /// 2026-11-01 this node made 3,148 such dials in 4 h — identities that had
    /// run on this machine's own port 8800, so each dial was a handshake with
    /// ITSELF — and the LAN box, holding the same identities at this machine's
    /// LAN address, opened 1,540 connections here that it closed at once (#155).
    ///
    /// Safe to act on: the handshake proved who answers at `address`, and a
    /// computer that controls an address can only make us forget pairs that
    /// point at it.
    pub(super) fn forget_address_answered_by_another(&mut self, peer: PeerId, address: &Multiaddr) {
        let left_the_table = self
            .swarm
            .behaviour_mut()
            .kademlia
            .remove_address(&peer, address)
            .is_some();
        let registry = forget_in_registry(&self.shared_state.peer_registry, peer, address);
        // And out of the peer cache, which merges what it stored into every save
        // and so brought the pair back at every start (#413).
        self.proven_wrong
            .note(peer, address, std::time::Instant::now());
        tracing::debug!(
            %peer,
            %address,
            left_the_table,
            registry,
            "Another computer answered at this address — forgetting it for this peer"
        );
    }

    /// A dial to `peer` failed: if nothing has reached it for
    /// [`stale_peers::UNREACHABLE_FOR`], it leaves Kademlia's table, so lookups
    /// stop dialling it and stop handing it to every node that asks (#413).
    /// Only a failure that reached the address or tried every one counts — a
    /// transport error or another computer answering — never a dial this node
    /// refused or abandoned. A connection puts the peer back (identify).
    ///
    /// [`stale_peers::UNREACHABLE_FOR`]: super::stale_peers::UNREACHABLE_FOR
    pub(super) fn leave_the_table_if_out_of_reach(
        &mut self,
        peer: PeerId,
        error: &libp2p::swarm::DialError,
        now: std::time::Instant,
    ) {
        use libp2p::swarm::DialError;
        if !matches!(
            error,
            DialError::Transport(_) | DialError::WrongPeerId { .. }
        ) || self.swarm.is_connected(&peer)
            || !self.last_reached.out_of_reach_long(&peer, now)
        {
            return;
        }
        if self
            .swarm
            .behaviour_mut()
            .kademlia
            .remove_peer(&peer)
            .is_some()
        {
            tracing::debug!(
                %peer,
                "Out of reach for an hour — leaving the routing table"
            );
        }
    }

    /// Who owns this address, according to our own peer registry?
    ///
    /// The fallback for a PEX address that names nobody, so the caller can run
    /// the already-connected and not-a-SwarmLLM-node gates on it anyway. Peers
    /// on a build that labels its PEX addresses make this unnecessary; it is
    /// what keeps a mixed-version swarm from re-dialling the peers it is
    /// already talking to.
    ///
    /// Compared against the address as the owner advertised it, ignoring any
    /// `/p2p/` suffix on our side so a labelled record still matches a bare
    /// address. A miss is not an error — it means a node we have never seen,
    /// which is exactly what PEX is for.
    fn peer_id_for_known_address(&self, addr: &Multiaddr) -> Option<libp2p::PeerId> {
        let wanted = addr.to_string();
        self.shared_state.peer_registry.iter().find_map(|entry| {
            let matches = entry.addresses.iter().any(|known| {
                known == &wanted
                    || known
                        .rsplit_once("/p2p/")
                        .is_some_and(|(head, _)| head == wanted)
            });
            if !matches {
                return None;
            }
            entry
                .peer_id_bytes
                .as_deref()
                .and_then(|b| libp2p::PeerId::from_bytes(b).ok())
        })
    }

    /// Handle PEX response — dial unknown peers from the exchanged address list,
    /// ONE dial per peer carrying every address it was listed at
    /// (`discovery::group_dials_by_peer`). Limits to 5 dials per response to
    /// prevent connection storms.
    ///
    /// It dialled one address at a time, and a peer's second address was
    /// refused as "already dialling" — so the first address listed decided the
    /// transport for the life of the connection. The Melbourne peer lists its
    /// public TCP address before its QUIC one, and every fresh node reached it
    /// over TCP (2026-10-23, `realdelay.sh`, 6 of 6): Cubic and slow start
    /// after idle instead of the BBR every QUIC connection runs (`arch-network.md`
    /// § "A burst is paced by the link"), and nothing later adds a QUIC
    /// connection, because Identify gives Kademlia only the connected address.
    /// One dial with every address RACES them (`concurrent_dial`, up to
    /// `dial_concurrency_factor` = 8 at once) and keeps the first to connect —
    /// QUIC's one-round-trip handshake beats TCP's three or more unless UDP is
    /// blocked, the Happy Eyeballs shape (RFC 8305; go-libp2p's dial ranker
    /// dials QUIC first for the same reason).
    pub(super) fn handle_pex_response(&mut self, peer_addrs: &[String]) {
        const MAX_PEX_DIALS: usize = 5;
        let listed = peer_addrs.iter().filter_map(|addr_str| {
            let addr = addr_str.parse::<Multiaddr>().ok()?;
            // SEC: Filter out private/link-local/loopback/CGN IPs to prevent SSRF
            if is_non_public_addr(addr_str) {
                tracing::debug!(addr = %addr_str, "PEX: skipping private/loopback address");
                return None;
            }
            Some(addr)
        });
        // Whose address is this? The gates below need the answer, and so does
        // the dial itself. A peer that has not yet been upgraded sends bare
        // addresses, which name nobody: ask our own registry who owns one
        // before falling through to an unconditional dial — that fallback is
        // the discovery path for a genuinely unknown node, and it is also what
        // re-dialled an already-connected peer on every PEX round (gotcha #405).
        let owned: Vec<(Option<libp2p::PeerId>, Multiaddr)> = listed
            .map(|addr| {
                let peer = target_peer_from_address(&addr)
                    .or_else(|| self.peer_id_for_known_address(&addr));
                (peer, addr)
            })
            .collect();

        let mut dialed = 0;
        for dial in crate::network::discovery::group_dials_by_peer(owned) {
            if dialed >= MAX_PEX_DIALS {
                break;
            }
            if let Some(pid) = &dial.peer {
                // An exchange can list THIS node: dialled, it is a handshake
                // with ourselves that both ends then refuse (156 in 4 h,
                // 2026-11-01).
                if pid == self.swarm.local_peer_id() {
                    continue;
                }
                // Skip a peer Identify has already shown is not SwarmLLM.
                // PEX hands out addresses without any claim about whose they
                // are, so this is the only thing stopping us re-dialling the
                // same foreign nodes every PEX round for the life of the
                // process — which is how five of them stayed in the peer list.
                if self.foreign_peers.contains(pid) {
                    continue;
                }
                if self.swarm.is_connected(pid) {
                    continue;
                }
            }

            // SEC: Do NOT call kademlia.add_address here — PEX-supplied
            // (PeerId, Multiaddr) pairs are unauthenticated and would let an
            // attacker poison the routing table for eclipse attacks. The dial
            // below triggers Noise + identify, and the identify handler adds
            // the verified address to Kademlia post-handshake.
            // Dial BY PEER ID with the addresses as hints, not by bare
            // address. `DialOpts::from(Multiaddr)` reports no peer id even
            // when the address carries `/p2p/`, so a bare-address dial is
            // invisible to `dial_checked` — and to libp2p's own per-peer
            // dedup. PEX is the path that hands out foreign addresses, so
            // it is the one that most needs the gate to see them.
            // `DisconnectedAndNotDialing` — libp2p's own default, and
            // stricter than the `Disconnected` that was here. `Disconnected`
            // asks only whether a connection is ESTABLISHED, so two dials
            // issued before either completes both pass, and a peer ends up
            // with as many connections as the per-peer cap allows. PEX is a
            // discovery path with no urgency: if a dial to this peer is
            // already in flight there is nothing a second one can add, and
            // the swarm's dial timeout bounds how long that lasts.
            //
            // The mDNS site deliberately keeps the weaker condition — see
            // the comment there; a LAN rediscovery has to be able to
            // override a stale bootstrap attempt.
            tracing::debug!(peer = ?dial.peer, addrs = ?dial.addrs, "PEX: dialling");
            let opts = match dial.peer {
                Some(pid) => libp2p::swarm::dial_opts::DialOpts::peer_id(pid)
                    .addresses(dial.addrs)
                    .condition(libp2p::swarm::dial_opts::PeerCondition::DisconnectedAndNotDialing)
                    .build(),
                // An address naming nobody is its own entry: dialled bare.
                None => match dial.addrs.into_iter().next() {
                    Some(addr) => addr.into(),
                    None => continue,
                },
            };
            if let Err(e) = self.dial_checked(opts, "pex") {
                tracing::debug!(error = %e, "PEX: failed to dial peer");
            } else {
                dialed += 1;
            }
        }
        if dialed > 0 {
            tracing::info!(count = dialed, "PEX: dialed new peers");
        }
    }

    // ── S5: DHT-based shard holder resolution ──

    /// Issue DHT provider queries for all shards of a model.
    /// Results arrive asynchronously via GetProviders events and are merged
    /// into the model_registry's bounded shard_holders cache.
    pub(super) fn handle_dht_provider_query(&mut self, model_id: &crate::types::ModelId) {
        // Dedup: skip if we already have pending queries for any shard of this model
        let already_querying = self
            .pending_provider_queries
            .values()
            .any(|sid| &sid.model_id == model_id);
        if already_querying {
            tracing::debug!(model = %model_id, "DHT query skipped — already querying this model");
            return;
        }

        let manifest = match self.shared_state.model_registry.get_manifest(model_id) {
            Some(m) => m,
            None => {
                tracing::debug!(model = %model_id, "DHT query skipped — manifest not found");
                return;
            }
        };

        let mut queried = 0;
        for shard_info in &manifest.shards {
            let shard_id = crate::types::ShardId {
                model_id: model_id.clone(),
                index: shard_info.index,
            };
            match crate::network::discovery::query_shard_providers(&mut self.swarm, &shard_id) {
                Ok(query_id) => {
                    self.pending_provider_queries.insert(query_id, shard_id);
                    queried += 1;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "DHT provider query failed");
                }
            }
        }

        if queried > 0 {
            tracing::info!(
                model = %model_id,
                shards_queried = queried,
                "Issued DHT provider queries for shard holders"
            );
        }

        // Cap pending queries to prevent unbounded growth
        if self.pending_provider_queries.len() > MAX_PENDING_PROVIDER_QUERIES {
            let excess = self.pending_provider_queries.len() - MAX_PENDING_PROVIDER_QUERIES;
            let keys: Vec<_> = self
                .pending_provider_queries
                .keys()
                .take(excess)
                .cloned()
                .collect();
            for k in keys {
                self.pending_provider_queries.remove(&k);
            }
        }
    }

    /// Handle DHT provider results — convert PeerIds to NodeIds and merge
    /// into the model_registry's bounded shard_holders cache.
    pub(super) fn handle_dht_providers_found(
        &mut self,
        query_id: libp2p::kad::QueryId,
        providers: &std::collections::HashSet<libp2p::PeerId>,
    ) {
        // NETWORKING_PLAN Phase 3 — relay-service discovery results: dial the
        // discovered relays instead of resolving shard holders.
        if self.pending_relay_provider_query == Some(query_id) {
            self.pending_relay_provider_query = None;
            self.handle_relay_providers_found(providers);
            return;
        }

        let shard_id = match self.pending_provider_queries.get(&query_id) {
            Some(sid) => sid.clone(),
            None => return, // Unknown query, ignore
        };

        let mut resolved = Vec::new();
        for peer_id in providers {
            // Try local reverse map first (fast)
            if let Some(node_id) = self.peer_to_node_id(peer_id) {
                resolved.push(node_id);
            } else if let Some(node_id) = crate::network::transport::peer_id_to_node_id(peer_id) {
                // Derive from PeerId directly (works for Ed25519 identity-hashed PeerIds)
                resolved.push(node_id);
            }
        }

        // Record the uncapped swarm-wide count even when we couldn't resolve
        // every PeerId — `providers.len()` is what the DHT reports, and the
        // prune redundancy_ratio needs the true count, not a cache-capped
        // undercount. `record_global_holder_count` overwrites, so later
        // responses for the same shard supersede earlier ones.
        self.shared_state
            .model_registry
            .record_global_holder_count(shard_id.clone(), providers.len() as u32);

        if !resolved.is_empty() {
            tracing::debug!(
                shard = ?shard_id,
                providers = resolved.len(),
                global = providers.len(),
                "Merging DHT providers into shard holders cache"
            );
            self.shared_state
                .model_registry
                .merge_dht_providers(&shard_id, &resolved);
        }
    }
}

#[cfg(test)]
mod proven_wrong_tests {
    use super::*;
    use libp2p::swarm::{ConnectionId, DialError, SwarmEvent};
    use std::sync::Arc;
    use tokio::sync::{mpsc, watch};

    fn manager() -> NetworkManager {
        let dir = tempfile::tempdir().unwrap().keep();
        let db = crate::storage::db::Database::open(&dir).unwrap();
        let config = crate::config::Config::default();
        let (state, _, _) = crate::daemon::SharedState::new(
            config.clone(),
            crate::identity::Identity::generate(),
            db,
            Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        let (command_tx, command_rx) = mpsc::channel(8);
        let (out_tx, _dispatched) = mpsc::channel(8);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let (_dht_tx, dht_rx) = mpsc::channel(8);
        NetworkManager::new(
            state.clone(),
            &state.identity,
            &config,
            command_rx,
            command_tx,
            out_tx,
            shutdown_rx,
            None,
            dht_rx,
        )
        .unwrap()
    }

    fn in_the_table(manager: &mut NetworkManager, peer: &PeerId) -> bool {
        manager
            .swarm
            .behaviour_mut()
            .kademlia
            .kbuckets()
            .any(|bucket| bucket.iter().any(|e| e.node.key.preimage() == peer))
    }

    fn registry_entry(peer: PeerId, addresses: &[&str]) -> crate::types::PeerInfo {
        crate::types::PeerInfo {
            node_id: crate::types::NodeId([7; 32]),
            addresses: addresses.iter().map(|a| a.to_string()).collect(),
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms: None,
            trust_score: 0.5,
            peer_id_bytes: Some(peer.to_bytes()),
            ack_srtt_ms: None,
            goodput_bytes_per_sec: None,
            goodput_samples: 0,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: true,
        }
    }

    /// A dial that reached another computer at an address forgets that address
    /// for the peer it meant — in Kademlia's table and in the registry the peer
    /// cache and peer exchange read — and nothing else (#155, 2026-11-01: an
    /// identity that once listened on this machine's port 8800 was dialled
    /// 3,148 times in 4 h, each a handshake with this node itself).
    #[tokio::test]
    async fn an_address_another_computer_answered_at_is_forgotten_for_the_peer_it_meant() {
        let mut manager = manager();
        let stale = PeerId::random();
        let answered = PeerId::random();
        let lan: Multiaddr = "/ip4/192.168.1.60/udp/8800/quic-v1".parse().unwrap();
        let elsewhere = "/ip4/198.51.100.9/tcp/8810";
        manager
            .swarm
            .behaviour_mut()
            .kademlia
            .add_address(&stale, lan.clone());
        manager.shared_state.peer_registry.insert(
            crate::types::NodeId([7; 32]),
            registry_entry(stale, &["/ip4/192.168.1.60/udp/8800/quic-v1", elsewhere]),
        );
        assert!(in_the_table(&mut manager, &stale));

        // The swarm names the address it dialled with the peer it meant.
        let error = DialError::WrongPeerId {
            obtained: answered,
            address: lan.clone().with_p2p(stale).unwrap(),
        };
        manager
            .handle_swarm_event(SwarmEvent::OutgoingConnectionError {
                connection_id: ConnectionId::new_unchecked(1),
                peer_id: Some(stale),
                error,
            })
            .await;

        assert!(
            !in_the_table(&mut manager, &stale),
            "its only address answered as someone else: the peer leaves the table"
        );
        let kept = manager
            .shared_state
            .peer_registry
            .get(&crate::types::NodeId([7; 32]))
            .unwrap()
            .addresses
            .clone();
        assert_eq!(
            kept,
            vec![elsewhere.to_string()],
            "only the address proven wrong goes"
        );
    }

    /// #413 (A): a pair another computer answered at leaves the PEER CACHE too
    /// — it merged the pair back into every save, so it returned at every start
    /// and was seeded into Kademlia again — and only that pair goes.
    #[tokio::test]
    async fn a_pair_another_computer_answered_at_leaves_the_peer_cache() {
        let mut manager = manager();
        let stale = PeerId::random();
        let lan: Multiaddr = "/ip4/192.168.1.60/udp/8800/quic-v1".parse().unwrap();
        let wrong_entry = format!("{lan}/p2p/{stale}");
        let other_entry = format!("/ip4/192.168.1.61/tcp/8810/p2p/{stale}");
        crate::network::peer_cache::save_peer_cache(
            &manager.shared_state.db,
            &[wrong_entry.clone(), other_entry.clone()],
        );
        assert!(manager.dialable_peer_cache().contains(&wrong_entry));

        manager
            .handle_swarm_event(SwarmEvent::OutgoingConnectionError {
                connection_id: ConnectionId::new_unchecked(1),
                peer_id: Some(stale),
                error: DialError::WrongPeerId {
                    obtained: PeerId::random(),
                    address: lan.clone().with_p2p(stale).unwrap(),
                },
            })
            .await;

        let dialable = manager.dialable_peer_cache();
        assert!(
            !dialable.contains(&wrong_entry),
            "not dialled from the cache again"
        );
        assert!(
            dialable.contains(&other_entry),
            "the peer elsewhere still is"
        );
        manager.save_peer_cache().await.unwrap();
        let stored = crate::network::peer_cache::load_peer_cache(&manager.shared_state.db);
        assert!(!stored.contains(&wrong_entry), "and not stored again");
        assert!(stored.contains(&other_entry));
    }

    /// #413 (B): a peer out of reach for an hour leaves Kademlia's table on its
    /// next failed dial; one reached within the hour stays, nothing goes in the
    /// node's first hour, and a dial this node abandoned proves nothing.
    #[tokio::test]
    async fn a_peer_out_of_reach_for_an_hour_leaves_the_routing_table() {
        use super::super::stale_peers::{LastReached, UNREACHABLE_FOR};
        let mut manager = manager();
        let started = std::time::Instant::now();
        manager.last_reached = LastReached::new(started);
        let gone = PeerId::random();
        let recent = PeerId::random();
        for peer in [gone, recent] {
            manager
                .swarm
                .behaviour_mut()
                .kademlia
                .add_address(&peer, "/ip4/203.0.113.7/udp/8800/quic-v1".parse().unwrap());
        }
        manager
            .last_reached
            .reached(recent, started + UNREACHABLE_FOR / 2);
        let failed = DialError::Transport(Vec::new());
        let later = started + UNREACHABLE_FOR;

        manager.leave_the_table_if_out_of_reach(gone, &DialError::Aborted, later);
        assert!(
            in_the_table(&mut manager, &gone),
            "an abandoned dial proves nothing"
        );
        manager.leave_the_table_if_out_of_reach(gone, &failed, started + UNREACHABLE_FOR / 2);
        assert!(
            in_the_table(&mut manager, &gone),
            "not in the node's first hour"
        );
        manager.leave_the_table_if_out_of_reach(recent, &failed, later);
        assert!(
            in_the_table(&mut manager, &recent),
            "reached within the hour"
        );
        manager.leave_the_table_if_out_of_reach(gone, &failed, later);
        assert!(
            !in_the_table(&mut manager, &gone),
            "out of reach for an hour"
        );
    }

    /// A dial that failed WITHOUT reaching anybody proves nothing about who is
    /// at the address: a timeout, a refusal, no addresses at all.
    #[test]
    fn only_a_handshake_with_another_computer_proves_the_pair_wrong() {
        let meant = PeerId::random();
        let address: Multiaddr = "/ip4/192.168.1.60/tcp/8810".parse().unwrap();
        let wrong = DialError::WrongPeerId {
            obtained: PeerId::random(),
            address: address.clone(),
        };
        assert_eq!(
            pair_proven_wrong(Some(meant), &wrong),
            Some((meant, &address))
        );
        assert_eq!(
            pair_proven_wrong(None, &wrong),
            None,
            "a bare dial meant nobody"
        );
        assert_eq!(
            pair_proven_wrong(Some(meant), &DialError::NoAddresses),
            None
        );
        assert_eq!(pair_proven_wrong(Some(meant), &DialError::Aborted), None);
    }

    #[test]
    fn a_registry_address_matches_with_or_without_its_peer_part() {
        let peer = PeerId::random();
        let address: Multiaddr = "/ip4/192.168.1.60/udp/8800/quic-v1".parse().unwrap();
        let with_peer = address.clone().with_p2p(peer).unwrap();
        assert!(is_this_address_of(&address.to_string(), peer, &with_peer));
        assert!(is_this_address_of(&with_peer.to_string(), peer, &address));
        assert!(!is_this_address_of(
            "/ip4/192.168.1.60/tcp/8810",
            peer,
            &address
        ));
        let someone_else = address.clone().with_p2p(PeerId::random()).unwrap();
        assert!(!is_this_address_of(
            &someone_else.to_string(),
            peer,
            &address
        ));
        assert!(!is_this_address_of("not an address", peer, &address));
    }
}
