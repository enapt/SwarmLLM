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

use crate::network::helpers::is_non_public_addr;

use super::{NetworkManager, MAX_PENDING_PROVIDER_QUERIES};

impl NetworkManager {
    /// Handle PEX response — dial unknown peers from the exchanged address list.
    /// Limits to 5 dials per response to prevent connection storms.
    pub(super) fn handle_pex_response(&mut self, peer_addrs: &[String]) {
        const MAX_PEX_DIALS: usize = 5;
        let mut dialed = 0;
        for addr_str in peer_addrs {
            if dialed >= MAX_PEX_DIALS {
                break;
            }
            if let Ok(addr) = addr_str.parse::<Multiaddr>() {
                // SEC: Filter out private/link-local/loopback/CGN IPs to prevent SSRF
                if is_non_public_addr(addr_str) {
                    tracing::debug!(addr = %addr_str, "PEX: skipping private/loopback address");
                    continue;
                }

                // Extract peer ID to check if already connected
                let maybe_peer_id = addr.iter().find_map(|proto| {
                    if let libp2p::multiaddr::Protocol::P2p(pid) = proto {
                        Some(pid)
                    } else {
                        None
                    }
                });

                // Skip a peer Identify has already shown is not SwarmLLM.
                // PEX hands out addresses without any claim about whose they
                // are, so this is the only thing stopping us re-dialling the
                // same foreign nodes every PEX round for the life of the
                // process — which is how five of them stayed in the peer list.
                if let Some(pid) = &maybe_peer_id {
                    if self.foreign_peers.contains(pid) {
                        continue;
                    }
                }
                // Skip if already connected
                if let Some(pid) = &maybe_peer_id {
                    if self.swarm.is_connected(pid) {
                        continue;
                    }
                }

                // SEC: Do NOT call kademlia.add_address here — PEX-supplied
                // (PeerId, Multiaddr) pairs are unauthenticated and would let an
                // attacker poison the routing table for eclipse attacks. The dial
                // below triggers Noise + identify, and the identify handler adds
                // the verified address to Kademlia post-handshake.
                if let Err(e) = self.swarm.dial(addr) {
                    tracing::debug!(error = %e, "PEX: failed to dial peer");
                } else {
                    dialed += 1;
                }
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
