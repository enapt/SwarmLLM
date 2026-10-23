//! DHT + PEX handlers — sibling of manager/mod.rs.
//!
//! `handle_pex_response` dials peers from the inbound exchanged-peers list.
//! `handle_dht_provider_query` issues GetProviders for every shard of a
//! model. `handle_dht_providers_found` merges resolved providers into the
//! model_registry's bounded shard_holders cache.
//!
//! State (`pending_provider_queries`, `swarm`, `peer_to_node_id`) lives on
//! `NetworkManager` in mod.rs; this file is a `pub(super) impl` block.

use libp2p::Multiaddr;

use crate::network::helpers::{is_non_public_addr, target_peer_from_address};

use super::{NetworkManager, MAX_PENDING_PROVIDER_QUERIES};

impl NetworkManager {
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
