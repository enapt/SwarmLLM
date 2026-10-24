// Copyright 2023 Protocol Labs.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

use std::{
    borrow::Borrow,
    collections::{HashMap, VecDeque},
    error::Error,
    hash::{Hash, Hasher},
    net::{self, IpAddr, SocketAddr, SocketAddrV4},
    ops::{Deref, DerefMut},
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use futures::{channel::oneshot, Future, StreamExt};
use futures_timer::Delay;
use igd_next::PortMappingProtocol;
use libp2p_core::{
    multiaddr,
    transport::{ListenerId, PortUse},
    Endpoint, Multiaddr,
};
use libp2p_swarm::{
    derive_prelude::PeerId, dummy, ConnectionDenied, ConnectionId, ExpiredListenAddr, FromSwarm,
    NetworkBehaviour, NewListenAddr, ToSwarm,
};

use crate::tokio::{is_addr_global, Gateway};

/// The duration in seconds of a port mapping on the gateway.
const MAPPING_DURATION: u32 = 3600;

/// Renew the Mapping every half of `MAPPING_DURATION` to avoid the port being unmapped.
const MAPPING_TIMEOUT: u64 = MAPPING_DURATION as u64 / 2;

/// Maximum number of retry attempts for failed mappings.
const MAX_RETRY_ATTEMPTS: u32 = 5;

/// Base delay in seconds for exponential backoff (will be multiplied by 2^retry_count).
const BASE_RETRY_DELAY_SECS: u64 = 30;

/// Maximum delay in seconds between retry attempts.
const MAX_RETRY_DELAY_SECS: u64 = 1800;

/// SwarmLLM patch (FUTURE_WORK #329): after no gateway was found, look again after
/// this long, doubling to `MAX_SEARCH_WAIT_SECS`. Upstream searched once, at start,
/// so UPnP turned on at the router while a node ran was used only after a restart.
const FIRST_SEARCH_WAIT_SECS: u64 = 120;
const MAX_SEARCH_WAIT_SECS: u64 = 1800;

/// How long to wait before the search after the one that just failed.
fn next_search_wait(wait: Duration) -> Duration {
    (wait * 2).min(Duration::from_secs(MAX_SEARCH_WAIT_SECS))
}

/// A [`Gateway`] Request.
#[derive(Debug)]
pub(crate) enum GatewayRequest {
    AddMapping { mapping: Mapping, duration: u32 },
    RemoveMapping(Mapping),
}

/// A [`Gateway`] event.
#[derive(Debug)]
pub(crate) enum GatewayEvent {
    /// Port was successfully mapped, on this external port (SwarmLLM patch: not
    /// always the internal one — `choice.rs`).
    Mapped {
        mapping: Mapping,
        external_port: u16,
    },
    /// There was a failure mapping port.
    MapFailure(Mapping, Box<dyn Error + Send + Sync + 'static>),
    /// Port was successfully removed.
    Removed(Mapping),
    /// There was a failure removing the mapped port.
    RemovalFailure(Mapping, Box<dyn Error + Send + Sync + 'static>),
}

/// Mapping of a Protocol and Port on the gateway.
#[derive(Debug, Clone)]
pub(crate) struct Mapping {
    pub(crate) listener_id: ListenerId,
    pub(crate) protocol: PortMappingProtocol,
    pub(crate) multiaddr: Multiaddr,
    pub(crate) internal_addr: SocketAddr,
}

impl Mapping {
    /// Given the input gateway address and the external port the router gave,
    /// calculate the open external `Multiaddr`.
    fn external_addr(&self, gateway_addr: IpAddr, external_port: u16) -> Multiaddr {
        let addr = match gateway_addr {
            net::IpAddr::V4(ip) => multiaddr::Protocol::Ip4(ip),
            net::IpAddr::V6(ip) => multiaddr::Protocol::Ip6(ip),
        };
        self.multiaddr
            .replace(0, |_| Some(addr))
            .and_then(|a| {
                a.replace(1, |p| match p {
                    multiaddr::Protocol::Tcp(_) => Some(multiaddr::Protocol::Tcp(external_port)),
                    multiaddr::Protocol::Udp(_) => Some(multiaddr::Protocol::Udp(external_port)),
                    _ => None,
                })
            })
            .expect("multiaddr should be valid")
    }
}

impl Hash for Mapping {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.listener_id.hash(state);
    }
}

impl PartialEq for Mapping {
    fn eq(&self, other: &Self) -> bool {
        self.listener_id == other.listener_id
    }
}

impl Eq for Mapping {}

impl Borrow<ListenerId> for Mapping {
    fn borrow(&self) -> &ListenerId {
        &self.listener_id
    }
}

/// Current state of a [`Mapping`].
#[derive(Debug)]
enum MappingState {
    /// Port mapping is inactive, will be requested or re-requested on the next iteration.
    Inactive,
    /// Port mapping/removal has been requested on the gateway.
    Pending {
        /// Number of times we've tried to map this port.
        retry_count: u32,
    },
    /// Port mapping is active with the inner timeout, on this external port.
    Active { renew: Delay, external_port: u16 },
    /// Port mapping failed with retry information.
    Failed {
        /// Number of times we've tried to map this port.
        retry_count: u32,
        /// When we should try again (None means no more retries).
        next_retry: Option<Delay>,
    },
    /// SwarmLLM patch (libp2p #6377, fixed upstream by PR 6459 in 0.7): the listen
    /// address went and its removal is on the way to the gateway. A reply to an
    /// earlier request changes nothing; the removal's own reply ends the entry.
    /// Upstream marked this `Pending` too, so a listener that came back while
    /// its removal was in flight lost its entry to the `Removed` reply and the
    /// next `Mapped` reply panicked ("mapping should exist").
    Removing {
        /// Removals asked for so far.
        attempts: u32,
    },
}

/// How many times a removal the router failed is asked for (upstream: without
/// limit or pause — each one a read of the router's table now).
const MAX_REMOVAL_ATTEMPTS: u32 = 3;

/// Current state of the UPnP [`Gateway`].
enum GatewayState {
    Searching(oneshot::Receiver<Result<Gateway, Box<dyn std::error::Error + Send + Sync>>>),
    Available(Gateway),
    /// No gateway answered; searched again when the delay fires.
    GatewayNotFound(Delay),
    NonRoutableGateway(IpAddr),
}

/// The event produced by `Behaviour`.
#[derive(Debug)]
pub enum Event {
    /// The multiaddress is reachable externally.
    NewExternalAddr {
        /// The local listen address that was mapped.
        local_addr: Multiaddr,
        /// The external address that is reachable.
        external_addr: Multiaddr,
    },
    /// The renewal of the multiaddress on the gateway failed.
    ExpiredExternalAddr {
        /// The local listen address that failed to renew.
        local_addr: Multiaddr,
        /// The external address that is no longer reachable.
        external_addr: Multiaddr,
    },
    /// The IGD gateway was not found.
    GatewayNotFound,
    /// The Gateway is not exposed directly to the public network.
    NonRoutableGateway,
}

/// A list of port mappings and its state.
#[derive(Debug, Default)]
struct MappingList(HashMap<Mapping, MappingState>);

impl Deref for MappingList {
    type Target = HashMap<Mapping, MappingState>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MappingList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl MappingList {
    /// SwarmLLM patch: the state of exactly this mapping — its listener AND its
    /// address. A reply about an address the listener no longer has is stale.
    fn state_of(&self, mapping: &Mapping) -> Option<&MappingState> {
        self.get_key_value(&mapping.listener_id)
            .filter(|(m, _)| {
                m.internal_addr == mapping.internal_addr && m.protocol == mapping.protocol
            })
            .map(|(_, state)| state)
    }

    /// SwarmLLM patch: store `state` under `mapping`, key included —
    /// `HashMap::insert` keeps the old key, so a listener whose address changed
    /// would go on being renewed at the old one.
    fn set(&mut self, mapping: Mapping, state: MappingState) {
        self.remove(&mapping.listener_id);
        self.insert(mapping, state);
    }

    /// Queue for renewal the current mapped ports on the `Gateway` that are expiring,
    /// and try to activate the inactive.
    fn renew(&mut self, gateway: &mut Gateway, cx: &mut Context<'_>) {
        for (mapping, state) in self.iter_mut() {
            match state {
                MappingState::Inactive => {
                    let duration = MAPPING_DURATION;
                    if let Err(err) = gateway.sender.try_send(GatewayRequest::AddMapping {
                        mapping: mapping.clone(),
                        duration,
                    }) {
                        tracing::debug!(
                            multiaddress=%mapping.multiaddr,
                            "could not request port mapping for multiaddress on the gateway: {}",
                            err
                        );
                    } else {
                        *state = MappingState::Pending { retry_count: 0 };
                    }
                }
                MappingState::Failed {
                    retry_count,
                    next_retry,
                } => {
                    if let Some(delay) = next_retry {
                        if Pin::new(delay).poll(cx).is_ready() {
                            let duration = MAPPING_DURATION;
                            if let Err(err) = gateway.sender.try_send(GatewayRequest::AddMapping {
                                mapping: mapping.clone(),
                                duration,
                            }) {
                                tracing::debug!(
                                    multiaddress=%mapping.multiaddr,
                                    retry_count=%retry_count,
                                    "could not retry port mapping for multiaddress on the gateway: {}",
                                    err
                                );
                            } else {
                                *state = MappingState::Pending {
                                    retry_count: *retry_count,
                                };
                            }
                        }
                    }
                }
                MappingState::Active { renew, .. } => {
                    if Pin::new(&mut *renew).poll(cx).is_ready() {
                        let duration = MAPPING_DURATION;
                        if let Err(err) = gateway.sender.try_send(GatewayRequest::AddMapping {
                            mapping: mapping.clone(),
                            duration,
                        }) {
                            tracing::debug!(
                                multiaddress=%mapping.multiaddr,
                                "could not request port mapping for multiaddress on the gateway: {}",
                                err
                            );
                        } else {
                            // SwarmLLM patch: a fired `Delay` stays ready, so every poll
                            // until the reply re-sent the renewal (a queue of them, each a
                            // read of the router's table now). Wait again; the reply
                            // resets it anyway.
                            *renew = Delay::new(Duration::from_secs(MAPPING_TIMEOUT));
                        }
                    }
                }
                MappingState::Pending { .. } | MappingState::Removing { .. } => {}
            }
        }
    }
}

/// A [`NetworkBehaviour`] for UPnP port mapping. Automatically tries to map the external port
/// to an internal address on the gateway on a [`FromSwarm::NewListenAddr`].
pub struct Behaviour {
    /// UPnP interface state.
    state: GatewayState,

    /// List of port mappings.
    mappings: MappingList,

    /// SwarmLLM patch: every listen address seen before a gateway was found.
    /// `mappings` holds ONE per listener (keyed by `listener_id`, and an insert
    /// keeps the first key), and which of a listener's addresses faces the
    /// router is known only once one answers — so they wait here (`place`).
    waiting: Vec<Mapping>,

    /// Pending behaviour events to be emitted.
    pending_events: VecDeque<Event>,

    /// How long the next search waits after one that found no gateway.
    search_wait: Duration,

    /// Swarm actions waiting their turn (SwarmLLM patch: a mapping that moved
    /// to another external port expires one address and confirms another).
    pending_actions: VecDeque<ToSwarm<Event, libp2p_swarm::THandlerInEvent<Self>>>,

    /// Every address this computer listens on, shared with the gateway task: a
    /// router row to any of them at our port is ours (`choice.rs`).
    own_ips: crate::tokio::OwnIps,
}

impl Default for Behaviour {
    fn default() -> Self {
        let own_ips = crate::tokio::OwnIps::default();
        Self {
            state: GatewayState::Searching(crate::tokio::search_gateway(Arc::clone(&own_ips))),
            mappings: Default::default(),
            waiting: Vec::new(),
            pending_events: VecDeque::new(),
            pending_actions: VecDeque::new(),
            search_wait: Duration::from_secs(FIRST_SEARCH_WAIT_SECS),
            own_ips,
        }
    }
}

impl NetworkBehaviour for Behaviour {
    type ConnectionHandler = dummy::ConnectionHandler;

    type ToSwarm = Event;

    fn handle_established_inbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _local_addr: &Multiaddr,
        _remote_addr: &Multiaddr,
    ) -> Result<libp2p_swarm::THandler<Self>, ConnectionDenied> {
        Ok(dummy::ConnectionHandler)
    }

    fn handle_established_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        _peer: PeerId,
        _addr: &Multiaddr,
        _role_override: Endpoint,
        _port_use: PortUse,
    ) -> Result<libp2p_swarm::THandler<Self>, libp2p_swarm::ConnectionDenied> {
        Ok(dummy::ConnectionHandler)
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        match event {
            FromSwarm::NewListenAddr(NewListenAddr {
                listener_id,
                addr: multiaddr,
            }) => {
                if let Some(ip) = first_ip(multiaddr) {
                    if let Ok(mut own) = self.own_ips.lock() {
                        if !own.contains(&ip) {
                            own.push(ip);
                        }
                    }
                }
                let Ok((addr, protocol)) = multiaddr_to_socketaddr_protocol(multiaddr.clone())
                else {
                    tracing::debug!("multiaddress not supported for UPnP {multiaddr}");
                    return;
                };

                if let Some((mapping, _state)) = self.mappings.iter().find(|(mapping, state)| {
                    matches!(state, MappingState::Active { .. })
                        && mapping.internal_addr.port() == addr.port()
                }) {
                    tracing::debug!(
                        multiaddress=%multiaddr,
                        mapped_multiaddress=%mapping.multiaddr,
                        "port from multiaddress is already mapped on the gateway"
                    );
                    return;
                }

                match &mut self.state {
                    // SwarmLLM patch: kept while no gateway is found too — a later
                    // search may find one (#329) — and every address, not the
                    // first of each listener: the one facing the router is picked
                    // when it answers (`place`).
                    GatewayState::Searching(_) | GatewayState::GatewayNotFound(_) => {
                        if !self.waiting.iter().any(|m| m.multiaddr == *multiaddr) {
                            self.waiting.push(Mapping {
                                listener_id,
                                protocol,
                                internal_addr: addr,
                                multiaddr: multiaddr.clone(),
                            });
                        }
                    }
                    GatewayState::Available(ref mut gateway) => {
                        // SwarmLLM patch: an address that is not the one the router
                        // was reached from may be its successor (DHCP, Wi-Fi to
                        // cable): read where it is reached from now.
                        if !faces_the_router(gateway.local_ip, addr.ip()) {
                            if let Some(ip) = crate::tokio::local_ip_towards(gateway.router) {
                                gateway.local_ip = Some(ip);
                            }
                        }
                        if !faces_the_router(gateway.local_ip, addr.ip()) {
                            tracing::debug!(
                                multiaddress=%multiaddr,
                                "not on the router's network; UPnP port mapping of multiaddress discarded"
                            );
                            return;
                        }
                        // SwarmLLM patch: one mapping per listener. Another address
                        // of a mapped listener (where the router-facing one is
                        // unknown) would ask for the same port again, and on a
                        // router that replaces, take it from the first. One whose
                        // removal is in flight is mapped again behind it.
                        if !matches!(
                            self.mappings.get(&listener_id),
                            None | Some(MappingState::Removing { .. })
                        ) {
                            tracing::debug!(
                                multiaddress=%multiaddr,
                                "this listener is mapped at another address already"
                            );
                            return;
                        }
                        let mapping = Mapping {
                            listener_id,
                            protocol,
                            internal_addr: addr,
                            multiaddr: multiaddr.clone(),
                        };

                        let duration = MAPPING_DURATION;
                        let state = match gateway.sender.try_send(GatewayRequest::AddMapping {
                            mapping: mapping.clone(),
                            duration,
                        }) {
                            Ok(()) => MappingState::Pending { retry_count: 0 },
                            Err(err) => {
                                tracing::debug!(
                                    multiaddress=%mapping.multiaddr,
                                    "could not request port mapping for multiaddress on the gateway: {}",
                                    err
                                );
                                // SwarmLLM patch: asked again at the next poll
                                // (upstream marked it Pending, and nothing ever
                                // answered).
                                MappingState::Inactive
                            }
                        };
                        self.mappings.set(mapping, state);
                    }
                    GatewayState::NonRoutableGateway(addr) => {
                        tracing::debug!(
                            multiaddress=%multiaddr,
                            network_gateway_ip=%addr,
                            "the network gateway is not exposed to the public network. /
                             UPnP port mapping of multiaddress discarded"
                        );
                    }
                };
            }
            FromSwarm::ExpiredListenAddr(ExpiredListenAddr { listener_id, addr }) => {
                if !matches!(self.state, GatewayState::Available(_)) {
                    // SwarmLLM patch: nothing is on a router yet; forget it, or a
                    // later search would map a port nothing listens on.
                    self.waiting
                        .retain(|m| m.listener_id != listener_id || m.multiaddr != *addr);
                } else if let GatewayState::Available(ref mut gateway) = &mut self.state {
                    // SwarmLLM patch: only when the address that went is the one
                    // mapped. A listener has several (one per interface) and only
                    // the router-facing one is mapped; upstream removed by listener
                    // alone, so WSL's address going took the LAN mapping with it.
                    let Some((mapping, state)) = self.mappings.get_key_value(&listener_id) else {
                        return;
                    };
                    if mapping.multiaddr != *addr || matches!(state, MappingState::Removing { .. })
                    {
                        return;
                    }
                    let mapping = mapping.clone();
                    // SwarmLLM patch: a mapped address that went is not reachable
                    // any more; upstream left it confirmed with the swarm.
                    if let MappingState::Active { external_port, .. } = *state {
                        let external = mapping.external_addr(gateway.external_addr, external_port);
                        self.pending_events.push_back(Event::ExpiredExternalAddr {
                            local_addr: mapping.multiaddr.clone(),
                            external_addr: external.clone(),
                        });
                        self.pending_actions
                            .push_back(ToSwarm::ExternalAddrExpired(external));
                    }
                    match gateway
                        .sender
                        .try_send(GatewayRequest::RemoveMapping(mapping.clone()))
                    {
                        Ok(()) => self
                            .mappings
                            .set(mapping, MappingState::Removing { attempts: 1 }),
                        Err(err) => {
                            tracing::debug!(
                                multiaddress=%mapping.multiaddr,
                                "could not request port removal for multiaddress on the gateway: {}",
                                err
                            );
                            self.mappings.remove(&listener_id);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn on_connection_handler_event(
        &mut self,
        _peer_id: PeerId,
        _connection_id: ConnectionId,
        event: libp2p_swarm::THandlerOutEvent<Self>,
    ) {
        libp2p_core::util::unreachable(event)
    }

    #[tracing::instrument(level = "trace", name = "NetworkBehaviour::poll", skip(self, cx))]
    fn poll(
        &mut self,
        cx: &mut Context<'_>,
    ) -> Poll<ToSwarm<Self::ToSwarm, libp2p_swarm::THandlerInEvent<Self>>> {
        if let Some(action) = self.pending_actions.pop_front() {
            return Poll::Ready(action);
        }

        // If there are pending addresses to be emitted we emit them.
        if let Some(event) = self.pending_events.pop_front() {
            return Poll::Ready(ToSwarm::GenerateEvent(event));
        }

        // Loop through the gateway state so that if it changes from `Searching` to `Available`
        // we poll the pending mapping requests.
        loop {
            match self.state {
                GatewayState::Searching(ref mut fut) => match Pin::new(fut).poll(cx) {
                    Poll::Ready(Ok(result)) => match result {
                        Ok(gateway) => {
                            if !is_addr_global(gateway.external_addr) {
                                self.state =
                                    GatewayState::NonRoutableGateway(gateway.external_addr);
                                tracing::debug!(
                                    gateway_address=%gateway.external_addr,
                                    "the gateway is not routable"
                                );
                                return Poll::Ready(ToSwarm::GenerateEvent(
                                    Event::NonRoutableGateway,
                                ));
                            }
                            // SwarmLLM patch: a listen address on another network
                            // (WSL's 10.255.255.254, a Docker bridge) cannot be
                            // forwarded to; upstream asked the router for it too.
                            for mapping in
                                place(std::mem::take(&mut self.waiting), gateway.local_ip)
                            {
                                self.mappings.set(mapping, MappingState::Inactive);
                            }
                            self.state = GatewayState::Available(gateway);
                        }
                        Err(err) => {
                            tracing::debug!("could not find gateway: {err}");
                            self.state =
                                GatewayState::GatewayNotFound(Delay::new(self.search_wait));
                            self.search_wait = next_search_wait(self.search_wait);
                            return Poll::Ready(ToSwarm::GenerateEvent(Event::GatewayNotFound));
                        }
                    },
                    Poll::Ready(Err(err)) => {
                        // The sender channel has been dropped. This typically indicates a shutdown
                        // process is underway.
                        tracing::debug!("sender has been dropped: {err}");
                        self.state = GatewayState::GatewayNotFound(Delay::new(self.search_wait));
                        self.search_wait = next_search_wait(self.search_wait);
                        return Poll::Ready(ToSwarm::GenerateEvent(Event::GatewayNotFound));
                    }
                    Poll::Pending => return Poll::Pending,
                },
                GatewayState::Available(ref mut gateway) => {
                    // Poll pending mapping requests. SwarmLLM patch: every reply
                    // waiting, not one a poll — a reply that returned nothing left
                    // the receiver un-polled, so the next one woke nobody.
                    while let Poll::Ready(Some(result)) = gateway.receiver.poll_next_unpin(cx) {
                        match result {
                            GatewayEvent::Mapped {
                                mapping,
                                external_port,
                            } => {
                                // SwarmLLM patch (libp2p #6377): no `expect` — a reply
                                // can outlive the address it was for.
                                let was = match self.mappings.state_of(&mapping) {
                                    None => {
                                        tracing::debug!(
                                            address=%mapping.internal_addr,
                                            protocol=%mapping.protocol,
                                            "mapped an address this listener no longer has; handing it back"
                                        );
                                        let _ = gateway
                                            .sender
                                            .try_send(GatewayRequest::RemoveMapping(mapping));
                                        continue;
                                    }
                                    Some(MappingState::Removing { .. }) => {
                                        tracing::debug!(
                                            address=%mapping.internal_addr,
                                            protocol=%mapping.protocol,
                                            "mapped as its address went; the removal follows"
                                        );
                                        continue;
                                    }
                                    Some(MappingState::Active { external_port, .. }) => {
                                        Some(*external_port)
                                    }
                                    Some(_) => None,
                                };
                                let external_multiaddr =
                                    mapping.external_addr(gateway.external_addr, external_port);
                                self.mappings.set(
                                    mapping.clone(),
                                    MappingState::Active {
                                        renew: Delay::new(Duration::from_secs(MAPPING_TIMEOUT)),
                                        external_port,
                                    },
                                );
                                match was {
                                    None => {
                                        self.pending_events.push_back(Event::NewExternalAddr {
                                            local_addr: mapping.multiaddr,
                                            external_addr: external_multiaddr.clone(),
                                        });
                                        tracing::debug!(
                                            address=%mapping.internal_addr,
                                            protocol=%mapping.protocol,
                                            external_port,
                                            "successfully mapped UPnP for protocol"
                                        );
                                        return Poll::Ready(ToSwarm::ExternalAddrConfirmed(
                                            external_multiaddr,
                                        ));
                                    }
                                    Some(was) if was == external_port => {
                                        tracing::debug!(
                                            address=%mapping.internal_addr,
                                            protocol=%mapping.protocol,
                                            "successfully renewed UPnP mapping for protocol"
                                        );
                                    }
                                    // SwarmLLM patch: another computer took the port, so
                                    // the renewal moved — the old address is not ours.
                                    Some(was) => {
                                        let old = mapping.external_addr(gateway.external_addr, was);
                                        self.pending_events.push_back(Event::ExpiredExternalAddr {
                                            local_addr: mapping.multiaddr.clone(),
                                            external_addr: old.clone(),
                                        });
                                        self.pending_events.push_back(Event::NewExternalAddr {
                                            local_addr: mapping.multiaddr,
                                            external_addr: external_multiaddr.clone(),
                                        });
                                        self.pending_actions.push_back(
                                            ToSwarm::ExternalAddrConfirmed(external_multiaddr),
                                        );
                                        return Poll::Ready(ToSwarm::ExternalAddrExpired(old));
                                    }
                                }
                            }
                            GatewayEvent::MapFailure(mapping, err) => {
                                let (retry_count, was_active) =
                                    match self.mappings.state_of(&mapping) {
                                        // A reply that outlived its address: nothing to do.
                                        None | Some(MappingState::Removing { .. }) => continue,
                                        Some(MappingState::Active { external_port, .. }) => {
                                            (0, Some(*external_port))
                                        }
                                        Some(
                                            MappingState::Pending { retry_count }
                                            | MappingState::Failed { retry_count, .. },
                                        ) => (*retry_count, None),
                                        Some(MappingState::Inactive) => (0, None),
                                    };

                                let new_retry_count = retry_count + 1;
                                let next_retry = if new_retry_count < MAX_RETRY_ATTEMPTS {
                                    let delay_secs = std::cmp::min(
                                        BASE_RETRY_DELAY_SECS
                                            .saturating_mul(2_u64.pow(retry_count)),
                                        MAX_RETRY_DELAY_SECS,
                                    );
                                    Some(Delay::new(Duration::from_secs(delay_secs)))
                                } else {
                                    tracing::warn!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        "giving up on UPnP mapping after {new_retry_count} attempts"
                                    );
                                    None
                                };

                                self.mappings.set(
                                    mapping.clone(),
                                    MappingState::Failed {
                                        retry_count: new_retry_count,
                                        next_retry,
                                    },
                                );

                                if let Some(external_port) = was_active {
                                    tracing::debug!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        "failed to remap UPnP mapped for protocol: {err}"
                                    );
                                    let external_multiaddr =
                                        mapping.external_addr(gateway.external_addr, external_port);
                                    self.pending_events.push_back(Event::ExpiredExternalAddr {
                                        local_addr: mapping.multiaddr,
                                        external_addr: external_multiaddr.clone(),
                                    });
                                    return Poll::Ready(ToSwarm::ExternalAddrExpired(
                                        external_multiaddr,
                                    ));
                                } else {
                                    tracing::debug!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        retry_count=%new_retry_count,
                                        "failed to map UPnP mapped for protocol: {err}"
                                    );
                                }
                            }
                            GatewayEvent::Removed(mapping) => {
                                // SwarmLLM patch: only a removal still wanted ends the
                                // entry — the listener may have been mapped again since
                                // (upstream: `expect`, and the panic of libp2p #6377).
                                if matches!(
                                    self.mappings.state_of(&mapping),
                                    Some(MappingState::Removing { .. })
                                ) {
                                    tracing::debug!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        "successfully removed UPnP mapping for protocol"
                                    );
                                    self.mappings.remove(&mapping.listener_id);
                                } else {
                                    tracing::debug!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        "removed a UPnP mapping its listener no longer needed"
                                    );
                                }
                            }
                            GatewayEvent::RemovalFailure(mapping, err) => {
                                tracing::debug!(
                                    address=%mapping.internal_addr,
                                    protocol=%mapping.protocol,
                                    "could not remove UPnP mapping for protocol: {err}"
                                );
                                // SwarmLLM patch: asked again up to `MAX_REMOVAL_ATTEMPTS`
                                // in all, not for ever; then the router's lease ends it.
                                let attempts = match self.mappings.state_of(&mapping) {
                                    Some(MappingState::Removing { attempts }) => *attempts,
                                    _ => continue,
                                };
                                if attempts < MAX_REMOVAL_ATTEMPTS
                                    && gateway
                                        .sender
                                        .try_send(GatewayRequest::RemoveMapping(mapping.clone()))
                                        .is_ok()
                                {
                                    self.mappings.set(
                                        mapping,
                                        MappingState::Removing {
                                            attempts: attempts + 1,
                                        },
                                    );
                                } else {
                                    tracing::debug!(
                                        address=%mapping.internal_addr,
                                        protocol=%mapping.protocol,
                                        "giving up removing the UPnP mapping; its lease ends it"
                                    );
                                    self.mappings.remove(&mapping.listener_id);
                                }
                            }
                        }
                    }

                    // Renew expired and request inactive mappings.
                    self.mappings.renew(gateway, cx);
                    return Poll::Pending;
                }
                GatewayState::GatewayNotFound(ref mut retry) => {
                    if Pin::new(retry).poll(cx).is_ready() {
                        tracing::debug!("searching for the network gateway again");
                        self.state = GatewayState::Searching(crate::tokio::search_gateway(
                            Arc::clone(&self.own_ips),
                        ));
                        continue;
                    }
                    return Poll::Pending;
                }
                _ => return Poll::Pending,
            }
        }
    }
}

/// The address an `/ip4` or `/ip6` multiaddress starts with.
fn first_ip(addr: &Multiaddr) -> Option<IpAddr> {
    match addr.iter().next()? {
        multiaddr::Protocol::Ip4(ip) => Some(IpAddr::V4(ip)),
        multiaddr::Protocol::Ip6(ip) => Some(IpAddr::V6(ip)),
        _ => None,
    }
}

/// May a listen address on `ip` be forwarded to by the router this computer
/// reaches from `local_ip`? Unknown → yes (upstream's behaviour).
fn faces_the_router(local_ip: Option<IpAddr>, ip: IpAddr) -> bool {
    local_ip.is_none_or(|local| local == ip)
}

/// The addresses seen before a gateway answered that are to be mapped: at most
/// one per listener, the one facing the router (the first, where that is unknown).
fn place(waiting: Vec<Mapping>, local_ip: Option<IpAddr>) -> Vec<Mapping> {
    let mut out: Vec<Mapping> = Vec::new();
    for mapping in waiting {
        if faces_the_router(local_ip, mapping.internal_addr.ip())
            && !out.iter().any(|m| m.listener_id == mapping.listener_id)
        {
            out.push(mapping);
        }
    }
    out
}

/// Extracts a [`SocketAddrV4`] and [`PortMappingProtocol`] from a given [`Multiaddr`].
///
/// Fails if the given [`Multiaddr`] does not begin with an IP
/// protocol encapsulating a TCP or UDP port.
fn multiaddr_to_socketaddr_protocol(
    addr: Multiaddr,
) -> Result<(SocketAddr, PortMappingProtocol), ()> {
    let mut iter = addr.into_iter();
    match iter.next() {
        // Idg only supports Ipv4.
        Some(multiaddr::Protocol::Ip4(ipv4)) if ipv4.is_private() => match iter.next() {
            Some(multiaddr::Protocol::Tcp(port)) => {
                return Ok((
                    SocketAddr::V4(SocketAddrV4::new(ipv4, port)),
                    PortMappingProtocol::TCP,
                ));
            }
            Some(multiaddr::Protocol::Udp(port)) => {
                return Ok((
                    SocketAddr::V4(SocketAddrV4::new(ipv4, port)),
                    PortMappingProtocol::UDP,
                ));
            }
            _ => {}
        },
        _ => {}
    }
    Err(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapping(addr: &str) -> Mapping {
        let multiaddr: Multiaddr = addr.parse().unwrap();
        let (internal_addr, protocol) =
            multiaddr_to_socketaddr_protocol(multiaddr.clone()).unwrap();
        Mapping {
            listener_id: ListenerId::next(),
            protocol,
            multiaddr,
            internal_addr,
        }
    }

    #[test]
    fn the_external_address_carries_the_port_the_router_gave() {
        let public: IpAddr = "171.97.115.37".parse().unwrap();
        let tcp = mapping("/ip4/192.168.1.52/tcp/8810");
        assert_eq!(
            tcp.external_addr(public, 8810).to_string(),
            "/ip4/171.97.115.37/tcp/8810"
        );
        assert_eq!(
            tcp.external_addr(public, 41_234).to_string(),
            "/ip4/171.97.115.37/tcp/41234"
        );
        let quic = mapping("/ip4/192.168.1.52/udp/8800/quic-v1");
        assert_eq!(
            quic.external_addr(public, 47_001).to_string(),
            "/ip4/171.97.115.37/udp/47001/quic-v1"
        );
    }

    #[test]
    fn a_router_that_did_not_answer_is_looked_for_again_ever_less_often() {
        let mut wait = Duration::from_secs(FIRST_SEARCH_WAIT_SECS);
        let mut waits = Vec::new();
        for _ in 0..6 {
            waits.push(wait.as_secs());
            wait = next_search_wait(wait);
        }
        assert_eq!(waits, [120, 240, 480, 960, 1800, 1800]);
    }

    #[test]
    fn only_an_address_on_the_routers_network_is_mapped() {
        let lan: IpAddr = "192.168.1.52".parse().unwrap();
        assert!(faces_the_router(Some(lan), lan));
        // WSL's DNS tunnel address is private too, and the router cannot reach it.
        assert!(!faces_the_router(
            Some(lan),
            "10.255.255.254".parse().unwrap()
        ));
        assert!(faces_the_router(None, "10.255.255.254".parse().unwrap()));
    }

    #[test]
    fn a_listeners_router_facing_address_is_mapped_whichever_came_first() {
        // One listener on 0.0.0.0 reports an address per interface, before the
        // router has answered. WSL's arriving first must not cost the LAN one.
        let lan: IpAddr = "192.168.1.52".parse().unwrap();
        let listener = ListenerId::next();
        let on = |addr: &str| Mapping {
            listener_id: listener,
            ..mapping(addr)
        };
        let waiting = vec![
            on("/ip4/10.255.255.254/tcp/8810"),
            on("/ip4/192.168.1.52/tcp/8810"),
        ];
        let placed = place(waiting.clone(), Some(lan));
        assert_eq!(placed.len(), 1);
        assert_eq!(placed[0].internal_addr.ip(), lan);
        // Router-facing address unknown: one per listener still, the first.
        let placed = place(waiting, None);
        assert_eq!(placed.len(), 1);
        assert_eq!(
            placed[0].internal_addr.ip(),
            "10.255.255.254".parse::<IpAddr>().unwrap()
        );
        // Another listener (the QUIC one) is placed on its own.
        let quic = mapping("/ip4/192.168.1.52/udp/8800/quic-v1");
        let placed = place(vec![on("/ip4/192.168.1.52/tcp/8810"), quic], Some(lan));
        assert_eq!(placed.len(), 2);
    }

    /// A behaviour whose gateway is two channels the test holds: what it asks
    /// the router for, and the router's replies.
    struct Rig {
        behaviour: Behaviour,
        requests: futures::channel::mpsc::Receiver<GatewayRequest>,
        replies: futures::channel::mpsc::Sender<GatewayEvent>,
    }

    impl Rig {
        fn new(local_ip: Option<&str>) -> Self {
            let (sender, requests) = futures::channel::mpsc::channel(10);
            let (replies, receiver) = futures::channel::mpsc::channel(10);
            let gateway = Gateway {
                sender,
                receiver,
                external_addr: "171.97.115.37".parse().unwrap(),
                router: "192.168.1.1:5555".parse().unwrap(),
                local_ip: local_ip.map(|ip| ip.parse().unwrap()),
            };
            Rig {
                behaviour: Behaviour {
                    state: GatewayState::Available(gateway),
                    mappings: Default::default(),
                    waiting: Vec::new(),
                    pending_events: VecDeque::new(),
                    pending_actions: VecDeque::new(),
                    search_wait: Duration::from_secs(FIRST_SEARCH_WAIT_SECS),
                    own_ips: Default::default(),
                },
                requests,
                replies,
            }
        }

        fn listen(&mut self, listener_id: ListenerId, addr: &str) {
            let addr: Multiaddr = addr.parse().unwrap();
            self.behaviour
                .on_swarm_event(FromSwarm::NewListenAddr(NewListenAddr {
                    listener_id,
                    addr: &addr,
                }));
        }

        fn unlisten(&mut self, listener_id: ListenerId, addr: &str) {
            let addr: Multiaddr = addr.parse().unwrap();
            self.behaviour
                .on_swarm_event(FromSwarm::ExpiredListenAddr(ExpiredListenAddr {
                    listener_id,
                    addr: &addr,
                }));
        }

        /// The requests sent to the gateway since the last call: "add" / "remove".
        fn asked(&mut self) -> Vec<(&'static str, Mapping)> {
            let mut out = Vec::new();
            while let Ok(req) = self.requests.try_recv() {
                out.push(match req {
                    GatewayRequest::AddMapping { mapping, .. } => ("add", mapping),
                    GatewayRequest::RemoveMapping(mapping) => ("remove", mapping),
                });
            }
            out
        }

        fn reply(&mut self, event: GatewayEvent) {
            self.replies.try_send(event).unwrap();
        }

        /// Poll until the behaviour has nothing more: (confirmed, expired) addresses.
        fn run(&mut self) -> (Vec<Multiaddr>, Vec<Multiaddr>) {
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            let (mut confirmed, mut expired) = (Vec::new(), Vec::new());
            while let Poll::Ready(action) = self.behaviour.poll(&mut cx) {
                match action {
                    ToSwarm::ExternalAddrConfirmed(a) => confirmed.push(a),
                    ToSwarm::ExternalAddrExpired(a) => expired.push(a),
                    _ => {}
                }
            }
            (confirmed, expired)
        }

        fn state(&self, listener_id: ListenerId) -> Option<&MappingState> {
            self.behaviour.mappings.get(&listener_id)
        }
    }

    const LAN_TCP: &str = "/ip4/192.168.1.52/tcp/8810";

    #[test]
    fn a_listener_back_before_its_removal_is_answered_is_mapped_again() {
        // libp2p #6377, first case (an interface flap): the address goes and
        // comes back while the removal is in flight. Upstream's `Removed` dropped
        // the entry and the next `Mapped` panicked ("mapping should exist").
        let mut rig = Rig::new(Some("192.168.1.52"));
        let l = ListenerId::next();
        rig.listen(l, LAN_TCP);
        let [(_, m)]: [_; 1] = rig.asked().try_into().unwrap();
        rig.reply(GatewayEvent::Mapped {
            mapping: m.clone(),
            external_port: 8810,
        });
        let (confirmed, _) = rig.run();
        assert_eq!(confirmed.len(), 1);

        rig.unlisten(l, LAN_TCP);
        rig.listen(l, LAN_TCP);
        let asked: Vec<_> = rig.asked().into_iter().map(|(what, _)| what).collect();
        assert_eq!(asked, ["remove", "add"]);
        let (_, expired) = rig.run();
        assert_eq!(expired.len(), 1, "the address that went is expired");

        // The gateway answers in order: the removal, then the new mapping.
        rig.reply(GatewayEvent::Removed(m.clone()));
        rig.reply(GatewayEvent::Mapped {
            mapping: m,
            external_port: 8810,
        });
        let (confirmed, _) = rig.run();
        assert_eq!(confirmed.len(), 1, "mapped again and said so");
        assert!(matches!(rig.state(l), Some(MappingState::Active { .. })));
    }

    #[test]
    fn a_listener_closing_with_several_addresses_removes_its_mapping_once() {
        // libp2p #6377, second case: one listener, an address per interface,
        // and the router-facing one unknown. Upstream mapped both (the second
        // taking the first's port on a router that replaces) and removed by
        // listener once per address, so the second `Removed` panicked.
        let mut rig = Rig::new(None);
        let l = ListenerId::next();
        rig.listen(l, LAN_TCP);
        rig.listen(l, "/ip4/10.255.255.254/tcp/8810");
        let asked = rig.asked();
        assert_eq!(asked.len(), 1, "one mapping per listener");
        let m = asked[0].1.clone();
        rig.reply(GatewayEvent::Mapped {
            mapping: m.clone(),
            external_port: 8810,
        });
        rig.run();

        // WSL's address going alone leaves the LAN mapping be (upstream removed
        // by listener, so it took the mapping with it).
        rig.unlisten(l, "/ip4/10.255.255.254/tcp/8810");
        assert!(rig.asked().is_empty());
        assert!(matches!(rig.state(l), Some(MappingState::Active { .. })));

        rig.listen(l, "/ip4/10.255.255.254/tcp/8810");
        rig.unlisten(l, LAN_TCP);
        rig.unlisten(l, "/ip4/10.255.255.254/tcp/8810");
        let asked: Vec<_> = rig.asked().into_iter().map(|(what, _)| what).collect();
        assert_eq!(asked, ["remove"]);
        rig.reply(GatewayEvent::Removed(m));
        rig.run();
        assert!(rig.state(l).is_none());
    }

    #[test]
    fn a_reply_that_outlived_its_address_confirms_nothing() {
        let mut rig = Rig::new(Some("192.168.1.52"));
        let l = ListenerId::next();
        rig.listen(l, LAN_TCP);
        let [(_, m)]: [_; 1] = rig.asked().try_into().unwrap();
        // The address goes before the router has answered the mapping.
        rig.unlisten(l, LAN_TCP);
        rig.reply(GatewayEvent::Mapped {
            mapping: m.clone(),
            external_port: 8810,
        });
        let (confirmed, _) = rig.run();
        assert!(confirmed.is_empty(), "{confirmed:?}");
        rig.reply(GatewayEvent::Removed(m));
        rig.run();
        assert!(rig.state(l).is_none());
    }

    #[test]
    fn a_removal_the_router_fails_is_asked_for_a_bounded_number_of_times() {
        let mut rig = Rig::new(Some("192.168.1.52"));
        let l = ListenerId::next();
        rig.listen(l, LAN_TCP);
        let [(_, m)]: [_; 1] = rig.asked().try_into().unwrap();
        rig.reply(GatewayEvent::Mapped {
            mapping: m.clone(),
            external_port: 8810,
        });
        rig.run();
        rig.unlisten(l, LAN_TCP);
        let mut removals = rig.asked().len();
        for _ in 0..10 {
            rig.reply(GatewayEvent::RemovalFailure(m.clone(), "refused".into()));
            rig.run();
            removals += rig.asked().len();
        }
        assert_eq!(removals, MAX_REMOVAL_ATTEMPTS as usize);
        assert!(rig.state(l).is_none(), "given up: the lease ends it");
    }
}
