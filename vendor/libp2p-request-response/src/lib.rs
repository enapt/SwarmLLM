// Copyright 2020 Parity Technologies (UK) Ltd.
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

//! Generic request/response protocols.
//!
//! ## General Usage
//!
//! The [`Behaviour`] struct is a [`NetworkBehaviour`] that implements a generic
//! request/response protocol or protocol family, whereby each request is
//! sent over a new substream on a connection. `Behaviour` is generic
//! over the actual messages being sent, which are defined in terms of a
//! [`Codec`]. Creating a request/response protocol thus amounts
//! to providing an implementation of this trait which can then be
//! given to [`Behaviour::with_codec`]. Further configuration options are
//! available via the [`Config`].
//!
//! Requests are sent using [`Behaviour::send_request`] and the
//! responses received as [`Message::Response`] via
//! [`Event::Message`].
//!
//! Responses are sent using [`Behaviour::send_response`] upon
//! receiving a [`Message::Request`] via
//! [`Event::Message`].
//!
//! ## Predefined codecs
//!
//! In case your message types implement [`serde::Serialize`] and [`serde::Deserialize`],
//! you can use two predefined behaviours:
//!
//! - [`cbor::Behaviour`] for CBOR-encoded messages
//! - [`json::Behaviour`] for JSON-encoded messages
//!
//! ## Protocol Families
//!
//! A single [`Behaviour`] instance can be used with an entire
//! protocol family that share the same request and response types.
//! For that purpose, [`Codec::Protocol`] is typically
//! instantiated with a sum type.
//!
//! ## Limited Protocol Support
//!
//! It is possible to only support inbound or outbound requests for
//! a particular protocol. This is achieved by instantiating `Behaviour`
//! with protocols using [`ProtocolSupport::Inbound`] or
//! [`ProtocolSupport::Outbound`]. Any subset of protocols of a protocol
//! family can be configured in this way. Such protocols will not be
//! advertised during inbound respectively outbound protocol negotiation
//! on the substreams.

#![cfg_attr(docsrs, feature(doc_cfg, doc_auto_cfg))]

#[cfg(feature = "cbor")]
pub mod cbor;
mod codec;
mod handler;
#[cfg(feature = "json")]
pub mod json;

use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt, io,
    sync::{atomic::AtomicU64, Arc},
    task::{Context, Poll},
    time::Duration,
};

pub use codec::Codec;
use futures::channel::oneshot;
use handler::Handler;
pub use handler::ProtocolSupport;
use libp2p_core::{transport::PortUse, ConnectedPoint, Endpoint, Multiaddr};
use libp2p_identity::PeerId;
use libp2p_swarm::{
    behaviour::{AddressChange, ConnectionClosed, DialFailure, FromSwarm},
    dial_opts::DialOpts,
    ConnectionDenied, ConnectionHandler, ConnectionId, DialError, NetworkBehaviour, NotifyHandler,
    PeerAddresses, THandler, THandlerInEvent, THandlerOutEvent, ToSwarm,
};
use smallvec::SmallVec;

use crate::handler::OutboundMessage;

/// An inbound request or response.
#[derive(Debug)]
pub enum Message<TRequest, TResponse, TChannelResponse = TResponse> {
    /// A request message.
    Request {
        /// The ID of this request.
        request_id: InboundRequestId,
        /// The request message.
        request: TRequest,
        /// The channel waiting for the response.
        ///
        /// If this channel is dropped instead of being used to send a response
        /// via [`Behaviour::send_response`], a [`Event::InboundFailure`]
        /// with [`InboundFailure::ResponseOmission`] is emitted.
        channel: ResponseChannel<TChannelResponse>,
    },
    /// A response message.
    Response {
        /// The ID of the request that produced this response.
        ///
        /// See [`Behaviour::send_request`].
        request_id: OutboundRequestId,
        /// The response message.
        response: TResponse,
    },
}

/// The events emitted by a request-response [`Behaviour`].
#[derive(Debug)]
pub enum Event<TRequest, TResponse, TChannelResponse = TResponse> {
    /// An incoming message (request or response).
    Message {
        /// The peer who sent the message.
        peer: PeerId,
        /// The connection used.
        connection_id: ConnectionId,
        /// The incoming message.
        message: Message<TRequest, TResponse, TChannelResponse>,
    },
    /// An outbound request failed.
    OutboundFailure {
        /// The peer to whom the request was sent.
        peer: PeerId,
        /// The connection used.
        connection_id: ConnectionId,
        /// The (local) ID of the failed request.
        request_id: OutboundRequestId,
        /// The error that occurred.
        error: OutboundFailure,
    },
    /// An inbound request failed.
    InboundFailure {
        /// The peer from whom the request was received.
        peer: PeerId,
        /// The connection used.
        connection_id: ConnectionId,
        /// The ID of the failed inbound request.
        request_id: InboundRequestId,
        /// The error that occurred.
        error: InboundFailure,
    },
    /// A response to an inbound request has been sent.
    ///
    /// When this event is received, the response has been flushed on
    /// the underlying transport connection.
    ResponseSent {
        /// The peer to whom the response was sent.
        peer: PeerId,
        /// The connection used.
        connection_id: ConnectionId,
        /// The ID of the inbound request whose response was sent.
        request_id: InboundRequestId,
    },
}

/// Possible failures occurring in the context of sending
/// an outbound request and receiving the response.
#[derive(Debug)]
pub enum OutboundFailure {
    /// The request could not be sent because a dialing attempt failed.
    DialFailure,
    /// The request timed out before a response was received.
    ///
    /// It is not known whether the request may have been
    /// received (and processed) by the remote peer.
    Timeout,
    /// The connection closed before a response was received.
    ///
    /// It is not known whether the request may have been
    /// received (and processed) by the remote peer.
    ConnectionClosed,
    /// The remote supports none of the requested protocols.
    UnsupportedProtocols,
    /// An IO failure happened on an outbound stream.
    Io(io::Error),
}

impl fmt::Display for OutboundFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutboundFailure::DialFailure => write!(f, "Failed to dial the requested peer"),
            OutboundFailure::Timeout => write!(f, "Timeout while waiting for a response"),
            OutboundFailure::ConnectionClosed => {
                write!(f, "Connection was closed before a response was received")
            }
            OutboundFailure::UnsupportedProtocols => {
                write!(f, "The remote supports none of the requested protocols")
            }
            OutboundFailure::Io(e) => write!(f, "IO error on outbound stream: {e}"),
        }
    }
}

impl std::error::Error for OutboundFailure {}

/// Possible failures occurring in the context of receiving an
/// inbound request and sending a response.
#[derive(Debug)]
pub enum InboundFailure {
    /// The inbound request timed out, either while reading the
    /// incoming request or before a response is sent, e.g. if
    /// [`Behaviour::send_response`] is not called in a
    /// timely manner.
    Timeout,
    /// The connection closed before a response could be send.
    ConnectionClosed,
    /// The local peer supports none of the protocols requested
    /// by the remote.
    UnsupportedProtocols,
    /// The local peer failed to respond to an inbound request
    /// due to the [`ResponseChannel`] being dropped instead of
    /// being passed to [`Behaviour::send_response`].
    ResponseOmission,
    /// An IO failure happened on an inbound stream.
    Io(io::Error),
}

impl fmt::Display for InboundFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InboundFailure::Timeout => {
                write!(f, "Timeout while receiving request or sending response")
            }
            InboundFailure::ConnectionClosed => {
                write!(f, "Connection was closed before a response could be sent")
            }
            InboundFailure::UnsupportedProtocols => write!(
                f,
                "The local peer supports none of the protocols requested by the remote"
            ),
            InboundFailure::ResponseOmission => write!(
                f,
                "The response channel was dropped without sending a response to the remote"
            ),
            InboundFailure::Io(e) => write!(f, "IO error on inbound stream: {e}"),
        }
    }
}

impl std::error::Error for InboundFailure {}

/// A channel for sending a response to an inbound request.
///
/// See [`Behaviour::send_response`].
#[derive(Debug)]
pub struct ResponseChannel<TResponse> {
    sender: oneshot::Sender<TResponse>,
}

impl<TResponse> ResponseChannel<TResponse> {
    /// Checks whether the response channel is still open, i.e.
    /// the `Behaviour` is still waiting for a
    /// a response to be sent via [`Behaviour::send_response`]
    /// and this response channel.
    ///
    /// If the response channel is no longer open then the inbound
    /// request timed out waiting for the response.
    pub fn is_open(&self) -> bool {
        !self.sender.is_canceled()
    }
}

/// The ID of an inbound request.
///
/// Note: [`InboundRequestId`]'s uniqueness is only guaranteed between
/// inbound requests of the same originating [`Behaviour`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InboundRequestId(u64);

impl fmt::Display for InboundRequestId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The ID of an outbound request.
///
/// Note: [`OutboundRequestId`]'s uniqueness is only guaranteed between
/// outbound requests of the same originating [`Behaviour`].
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OutboundRequestId(u64);

impl fmt::Display for OutboundRequestId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The configuration for a `Behaviour` protocol.
#[derive(Debug, Clone)]
pub struct Config {
    request_timeout: Duration,
    max_concurrent_streams: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(10),
            max_concurrent_streams: 100,
        }
    }
}

impl Config {
    /// Sets the timeout for inbound and outbound requests.
    #[deprecated(note = "Use `Config::with_request_timeout` for one-liner constructions.")]
    pub fn set_request_timeout(&mut self, v: Duration) -> &mut Self {
        self.request_timeout = v;
        self
    }

    /// Sets the timeout for inbound and outbound requests.
    pub fn with_request_timeout(mut self, v: Duration) -> Self {
        self.request_timeout = v;
        self
    }

    /// Sets the upper bound for the number of concurrent inbound + outbound streams.
    pub fn with_max_concurrent_streams(mut self, num_streams: usize) -> Self {
        self.max_concurrent_streams = num_streams;
        self
    }
}

/// A request/response protocol for some message codec.
pub struct Behaviour<TCodec>
where
    TCodec: Codec + Clone + Send + 'static,
{
    /// The supported inbound protocols.
    inbound_protocols: SmallVec<[TCodec::Protocol; 2]>,
    /// The supported outbound protocols.
    outbound_protocols: SmallVec<[TCodec::Protocol; 2]>,
    /// The next (local) request ID.
    next_outbound_request_id: OutboundRequestId,
    /// The next (inbound) request ID.
    next_inbound_request_id: Arc<AtomicU64>,
    /// The protocol configuration.
    config: Config,
    /// The protocol codec for reading and writing requests and responses.
    codec: TCodec,
    /// Pending events to return from `poll`.
    pending_events:
        VecDeque<ToSwarm<Event<TCodec::Request, TCodec::Response>, OutboundMessage<TCodec>>>,
    /// The currently connected peers, their pending outbound and inbound responses and their
    /// known, reachable addresses, if any.
    connected: HashMap<PeerId, SmallVec<[Connection; 2]>>,
    /// Externally managed addresses via `add_address` and `remove_address`.
    addresses: PeerAddresses,
    /// Requests that have not yet been sent and are waiting for a connection
    /// to be established.
    pending_outbound_requests: HashMap<PeerId, SmallVec<[OutboundMessage<TCodec>; 10]>>,
}

impl<TCodec> Behaviour<TCodec>
where
    TCodec: Codec + Default + Clone + Send + 'static,
{
    /// Creates a new `Behaviour` for the given protocols and configuration, using [`Default`] to
    /// construct the codec.
    pub fn new<I>(protocols: I, cfg: Config) -> Self
    where
        I: IntoIterator<Item = (TCodec::Protocol, ProtocolSupport)>,
    {
        Self::with_codec(TCodec::default(), protocols, cfg)
    }
}

impl<TCodec> Behaviour<TCodec>
where
    TCodec: Codec + Clone + Send + 'static,
{
    /// Creates a new `Behaviour` for the given
    /// protocols, codec and configuration.
    pub fn with_codec<I>(codec: TCodec, protocols: I, cfg: Config) -> Self
    where
        I: IntoIterator<Item = (TCodec::Protocol, ProtocolSupport)>,
    {
        let mut inbound_protocols = SmallVec::new();
        let mut outbound_protocols = SmallVec::new();
        for (p, s) in protocols {
            if s.inbound() {
                inbound_protocols.push(p.clone());
            }
            if s.outbound() {
                outbound_protocols.push(p.clone());
            }
        }
        Behaviour {
            inbound_protocols,
            outbound_protocols,
            next_outbound_request_id: OutboundRequestId(1),
            next_inbound_request_id: Arc::new(AtomicU64::new(1)),
            config: cfg,
            codec,
            pending_events: VecDeque::new(),
            connected: HashMap::new(),
            pending_outbound_requests: HashMap::new(),
            addresses: PeerAddresses::default(),
        }
    }

    /// Initiates sending a request.
    ///
    /// If the targeted peer is currently not connected, a dialing
    /// attempt is initiated and the request is sent as soon as a
    /// connection is established.
    ///
    /// > **Note**: In order for such a dialing attempt to succeed,
    /// > the `RequestResponse` protocol must either be embedded
    /// > in another `NetworkBehaviour` that provides peer and
    /// > address discovery, or known addresses of peers must be
    /// > managed via [`libp2p_swarm::Swarm::add_peer_address`].
    /// > Addresses are automatically removed when dial attempts
    /// > to them fail.
    /// > Alternatively, [`Behaviour::send_request_with_addresses`]
    /// > can be used.
    pub fn send_request(&mut self, peer: &PeerId, request: TCodec::Request) -> OutboundRequestId {
        self.send_request_with_addresses(peer, request, Vec::new())
    }

    /// Like [`Behaviour::send_request`], but additionally using the provided addresses
    /// if a connection needs to be established.
    pub fn send_request_with_addresses(
        &mut self,
        peer: &PeerId,
        request: TCodec::Request,
        addresses: Vec<Multiaddr>,
    ) -> OutboundRequestId {
        let request_id = self.next_outbound_request_id();
        let request = OutboundMessage {
            request_id,
            request,
            protocols: self.outbound_protocols.clone(),
        };

        if let Some(request) = self.try_send_request(peer, request) {
            self.pending_events.push_back(ToSwarm::Dial {
                opts: DialOpts::peer_id(*peer)
                    .addresses(addresses)
                    .extend_addresses_through_behaviour()
                    .build(),
            });
            self.pending_outbound_requests
                .entry(*peer)
                .or_default()
                .push(request);
        }

        request_id
    }

    /// Initiates sending a response to an inbound request.
    ///
    /// If the [`ResponseChannel`] is already closed due to a timeout or the
    /// connection being closed, the response is returned as an `Err` for
    /// further handling. Once the response has been successfully sent on the
    /// corresponding connection, [`Event::ResponseSent`] is
    /// emitted. In all other cases [`Event::InboundFailure`]
    /// will be or has been emitted.
    ///
    /// The provided `ResponseChannel` is obtained from an inbound
    /// [`Message::Request`].
    pub fn send_response(
        &mut self,
        ch: ResponseChannel<TCodec::Response>,
        rs: TCodec::Response,
    ) -> Result<(), TCodec::Response> {
        ch.sender.send(rs)
    }

    /// Adds a known address for a peer that can be used for
    /// dialing attempts by the `Swarm`, i.e. is returned
    /// by [`NetworkBehaviour::handle_pending_outbound_connection`].
    ///
    /// Addresses added in this way are only removed by `remove_address`.
    ///
    /// Returns true if the address was added, false otherwise (i.e. if the
    /// address is already in the list).
    #[deprecated(note = "Use `Swarm::add_peer_address` instead.")]
    pub fn add_address(&mut self, peer: &PeerId, address: Multiaddr) -> bool {
        self.addresses.add(*peer, address)
    }

    /// Removes an address of a peer previously added via [`Behaviour::add_address`].
    #[deprecated(note = "Will be removed with the next breaking release and won't be replaced.")]
    pub fn remove_address(&mut self, peer: &PeerId, address: &Multiaddr) {
        self.addresses.remove(peer, address);
    }

    /// Checks whether a peer is currently connected.
    pub fn is_connected(&self, peer: &PeerId) -> bool {
        if let Some(connections) = self.connected.get(peer) {
            !connections.is_empty()
        } else {
            false
        }
    }

    /// Checks whether an outbound request to the peer with the provided
    /// [`PeerId`] initiated by [`Behaviour::send_request`] is still
    /// pending, i.e. waiting for a response.
    pub fn is_pending_outbound(&self, peer: &PeerId, request_id: &OutboundRequestId) -> bool {
        // Check if request is already sent on established connection.
        let est_conn = self
            .connected
            .get(peer)
            .map(|cs| {
                cs.iter()
                    .any(|c| c.pending_outbound_responses.contains(request_id))
            })
            .unwrap_or(false);
        // Check if request is still pending to be sent.
        let pen_conn = self
            .pending_outbound_requests
            .get(peer)
            .map(|rps| rps.iter().any(|rp| rp.request_id == *request_id))
            .unwrap_or(false);

        est_conn || pen_conn
    }

    /// Checks whether an inbound request from the peer with the provided
    /// [`PeerId`] is still pending, i.e. waiting for a response by the local
    /// node through [`Behaviour::send_response`].
    pub fn is_pending_inbound(&self, peer: &PeerId, request_id: &InboundRequestId) -> bool {
        self.connected
            .get(peer)
            .map(|cs| {
                cs.iter()
                    .any(|c| c.pending_inbound_responses.contains(request_id))
            })
            .unwrap_or(false)
    }

    /// Returns the next outbound request ID.
    fn next_outbound_request_id(&mut self) -> OutboundRequestId {
        let request_id = self.next_outbound_request_id;
        self.next_outbound_request_id.0 += 1;
        request_id
    }

    /// Tries to send a request by queueing an appropriate event to be
    /// emitted to the `Swarm`. If the peer is not currently connected,
    /// the given request is return unchanged.
    fn try_send_request(
        &mut self,
        peer: &PeerId,
        request: OutboundMessage<TCodec>,
    ) -> Option<OutboundMessage<TCodec>> {
        if let Some(connections) = self.connected.get_mut(peer) {
            if connections.is_empty() {
                return Some(request);
            }
            // SwarmLLM patch: prefer a DIRECT connection over a relayed
            // (`/p2p-circuit`) one.
            //
            // Upstream round-robins blindly over every connection to the peer
            // (`request_id % connections.len()`). That is wrong for us in two
            // ways once a peer can have more than one connection:
            //
            //  1. A relay circuit and a direct connection routinely coexist —
            //     that is exactly how DCUtR upgrades work (it dials the direct
            //     connection while the relayed one is still open). Blind
            //     round-robin would keep sending half of all requests over the
            //     circuit even after a direct path exists, inheriting the
            //     circuit's unreliability (and its byte/duration caps) for no
            //     reason.
            //  2. Relayed substreams are the flaky path for request_response in
            //     the first place (libp2p/rust-libp2p#3034): the responder can
            //     close the stream before the response traverses the relay.
            //
            //  3. Round-robin is wrong even among EQUIVALENT connections.
            //     Observed live 2026-07-25: raising the per-peer connection cap
            //     (needed so DCUtR can hold a relayed and a direct connection at
            //     once) also permits redundant connections to the SAME endpoint —
            //     up to the cap, routinely. Spreading requests across those means
            //     a single half-open one silently eats its share, which is the
            //     original bug the cap of 1 was hiding rather than fixing.
            //
            // So: pick the NEWEST direct connection, falling back to the newest
            // connection of any kind. Newest is the right choice on every axis
            // here — a half-open connection is almost always an older one that
            // died quietly, and DCUtR's upgraded direct connection is by
            // definition the newest, so it wins automatically. Load-spreading
            // across connections to a single peer was never a real benefit;
            // they share a path and usually an endpoint.
            //
            //  4. Newest is NOT sufficient on its own. Observed live 2026-07-26:
            //     a connection that had just served three consecutive requests
            //     was passed over for a NEWER direct connection that silently
            //     swallowed every send — no response, no `OutboundFailure` —
            //     until the application's own 10s acknowledgement timeout gave
            //     up. Age is a heuristic, and this is the case where it points
            //     the wrong way.
            //
            //     `pending_outbound_responses` is the liveness signal already to
            //     hand: it is inserted on send and removed when the response
            //     arrives, so a connection that is answering drains it while a
            //     half-open one only accumulates. Preferring the direct
            //     connection with the FEWEST un-answered requests therefore
            //     steers away from a dead path after the first failure, instead
            //     of re-picking it every time. Ties break toward the newest,
            //     which preserves the DCUtR behaviour above.
            //
            // `connected` is append-ordered (`push` on establish), so a larger
            // index is more recently established.
            let best_direct = connections
                .iter()
                .enumerate()
                .filter(|(_, c)| !connection_is_relayed(c))
                .min_by_key(|(i, c)| {
                    (
                        c.pending_outbound_responses.len(),
                        std::cmp::Reverse(*i),
                    )
                })
                .map(|(i, _)| i);
            let ix = best_direct.unwrap_or(connections.len() - 1);
            let conn = &mut connections[ix];
            conn.pending_outbound_responses.insert(request.request_id);
            self.pending_events.push_back(ToSwarm::NotifyHandler {
                peer_id: *peer,
                handler: NotifyHandler::One(conn.id),
                event: request,
            });
            None
        } else {
            Some(request)
        }
    }

    /// Remove pending outbound response for the given peer and connection.
    ///
    /// Returns `true` if the provided connection to the given peer is still
    /// alive and the [`OutboundRequestId`] was previously present and is now removed.
    /// Returns `false` otherwise.
    fn remove_pending_outbound_response(
        &mut self,
        peer: &PeerId,
        connection_id: ConnectionId,
        request: OutboundRequestId,
    ) -> bool {
        self.get_connection_mut(peer, connection_id)
            .map(|c| c.pending_outbound_responses.remove(&request))
            .unwrap_or(false)
    }

    /// Remove pending inbound response for the given peer and connection.
    ///
    /// Returns `true` if the provided connection to the given peer is still
    /// alive and the [`InboundRequestId`] was previously present and is now removed.
    /// Returns `false` otherwise.
    fn remove_pending_inbound_response(
        &mut self,
        peer: &PeerId,
        connection_id: ConnectionId,
        request: InboundRequestId,
    ) -> bool {
        self.get_connection_mut(peer, connection_id)
            .map(|c| c.pending_inbound_responses.remove(&request))
            .unwrap_or(false)
    }

    /// Returns a mutable reference to the connection in `self.connected`
    /// corresponding to the given [`PeerId`] and [`ConnectionId`].
    fn get_connection_mut(
        &mut self,
        peer: &PeerId,
        connection_id: ConnectionId,
    ) -> Option<&mut Connection> {
        self.connected
            .get_mut(peer)
            .and_then(|connections| connections.iter_mut().find(|c| c.id == connection_id))
    }

    fn on_address_change(
        &mut self,
        AddressChange {
            peer_id,
            connection_id,
            new,
            ..
        }: AddressChange,
    ) {
        let new_address = match new {
            ConnectedPoint::Dialer { address, .. } => Some(address.clone()),
            ConnectedPoint::Listener { .. } => None,
        };
        let connections = self
            .connected
            .get_mut(&peer_id)
            .expect("Address change can only happen on an established connection.");

        let connection = connections
            .iter_mut()
            .find(|c| c.id == connection_id)
            .expect("Address change can only happen on an established connection.");
        connection.remote_address = new_address;
    }

    fn on_connection_closed(
        &mut self,
        ConnectionClosed {
            peer_id,
            connection_id,
            remaining_established,
            ..
        }: ConnectionClosed,
    ) {
        let connections = self
            .connected
            .get_mut(&peer_id)
            .expect("Expected some established connection to peer before closing.");

        let connection = connections
            .iter()
            .position(|c| c.id == connection_id)
            .map(|p: usize| connections.remove(p))
            .expect("Expected connection to be established before closing.");

        // SwarmLLM patch: trust `remaining_established` and clean up, rather
        // than asserting the two views agree.
        //
        // Upstream asserts this bookkeeping matches the swarm's. Observed
        // diverging once on 2026-08-05 (debug build, during a burst of four
        // concurrent requests immediately after `PEX: dialed new peers
        // count=4`): the swarm reported no connections remaining while this
        // behaviour still held entries. The `debug_assert` then panicked a
        // tokio worker and the supervisor took the whole daemon down.
        //
        // Release builds compile the assertion out, so users never saw the
        // panic — but they got the silent half of the same bug: the `if` below
        // was false, so the peer's entry was never removed, leaving a record of
        // connections that no longer exist. That can only mislead a later send.
        //
        // `remaining_established` comes from the swarm, which is the authority
        // on what is actually open, so it wins. Kept as a warning rather than
        // dropped in silence: the divergence itself is worth knowing about, and
        // one observation is not enough to say what causes it.
        if remaining_established == 0 && !connections.is_empty() {
            tracing::warn!(
                %peer_id,
                stale = connections.len(),
                "request_response: swarm reports no connections left but this \
                 behaviour still held some — dropping the stale entries"
            );
            connections.clear();
        }
        if connections.is_empty() {
            self.connected.remove(&peer_id);
        }

        for request_id in connection.pending_inbound_responses {
            self.pending_events
                .push_back(ToSwarm::GenerateEvent(Event::InboundFailure {
                    peer: peer_id,
                    connection_id,
                    request_id,
                    error: InboundFailure::ConnectionClosed,
                }));
        }

        for request_id in connection.pending_outbound_responses {
            self.pending_events
                .push_back(ToSwarm::GenerateEvent(Event::OutboundFailure {
                    peer: peer_id,
                    connection_id,
                    request_id,
                    error: OutboundFailure::ConnectionClosed,
                }));
        }
    }

    fn on_dial_failure(
        &mut self,
        DialFailure {
            peer_id,
            connection_id,
            error,
        }: DialFailure,
    ) {
        if let DialError::DialPeerConditionFalse(_) = error {
            // Dial-condition fails because there is already another ongoing dial.
            return;
        }
        if let Some(peer) = peer_id {
            // If there are pending outgoing requests when a dial failure occurs,
            // it is implied that we are not connected to the peer, since pending
            // outgoing requests are drained when a connection is established and
            // only created when a peer is not connected when a request is made.
            // Thus these requests must be considered failed, even if there is
            // another, concurrent dialing attempt ongoing.
            if let Some(pending) = self.pending_outbound_requests.remove(&peer) {
                for request in pending {
                    self.pending_events
                        .push_back(ToSwarm::GenerateEvent(Event::OutboundFailure {
                            peer,
                            connection_id,
                            request_id: request.request_id,
                            error: OutboundFailure::DialFailure,
                        }));
                }
            }
        }
    }

    /// Preloads a new [`Handler`] with requests that are
    /// waiting to be sent to the newly connected peer.
    fn preload_new_handler(
        &mut self,
        handler: &mut Handler<TCodec>,
        peer: PeerId,
        connection_id: ConnectionId,
        remote_address: Option<Multiaddr>,
    ) {
        let mut connection = Connection::new(connection_id, remote_address);

        if let Some(pending_requests) = self.pending_outbound_requests.remove(&peer) {
            for request in pending_requests {
                connection
                    .pending_outbound_responses
                    .insert(request.request_id);
                handler.on_behaviour_event(request);
            }
        }

        self.connected.entry(peer).or_default().push(connection);
    }
}

impl<TCodec> NetworkBehaviour for Behaviour<TCodec>
where
    TCodec: Codec + Send + Clone + 'static,
{
    type ConnectionHandler = Handler<TCodec>;
    type ToSwarm = Event<TCodec::Request, TCodec::Response>;

    fn handle_established_inbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        _: &Multiaddr,
        _: &Multiaddr,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        let mut handler = Handler::new(
            self.inbound_protocols.clone(),
            self.codec.clone(),
            self.config.request_timeout,
            self.next_inbound_request_id.clone(),
            self.config.max_concurrent_streams,
        );

        self.preload_new_handler(&mut handler, peer, connection_id, None);

        Ok(handler)
    }

    fn handle_pending_outbound_connection(
        &mut self,
        _connection_id: ConnectionId,
        maybe_peer: Option<PeerId>,
        _addresses: &[Multiaddr],
        _effective_role: Endpoint,
    ) -> Result<Vec<Multiaddr>, ConnectionDenied> {
        let Some(peer) = maybe_peer else {
            return Ok(vec![]);
        };

        let mut addresses = Vec::new();
        if let Some(connections) = self.connected.get(&peer) {
            addresses.extend(connections.iter().filter_map(|c| c.remote_address.clone()))
        }

        let cached_addrs = self.addresses.get(&peer);
        addresses.extend(cached_addrs);

        Ok(addresses)
    }

    fn handle_established_outbound_connection(
        &mut self,
        connection_id: ConnectionId,
        peer: PeerId,
        remote_address: &Multiaddr,
        _: Endpoint,
        _: PortUse,
    ) -> Result<THandler<Self>, ConnectionDenied> {
        let mut handler = Handler::new(
            self.inbound_protocols.clone(),
            self.codec.clone(),
            self.config.request_timeout,
            self.next_inbound_request_id.clone(),
            self.config.max_concurrent_streams,
        );

        self.preload_new_handler(
            &mut handler,
            peer,
            connection_id,
            Some(remote_address.clone()),
        );

        Ok(handler)
    }

    fn on_swarm_event(&mut self, event: FromSwarm) {
        self.addresses.on_swarm_event(&event);
        match event {
            FromSwarm::ConnectionEstablished(_) => {}
            FromSwarm::ConnectionClosed(connection_closed) => {
                self.on_connection_closed(connection_closed)
            }
            FromSwarm::AddressChange(address_change) => self.on_address_change(address_change),
            FromSwarm::DialFailure(dial_failure) => self.on_dial_failure(dial_failure),
            _ => {}
        }
    }

    fn on_connection_handler_event(
        &mut self,
        peer: PeerId,
        connection_id: ConnectionId,
        event: THandlerOutEvent<Self>,
    ) {
        match event {
            handler::Event::Response {
                request_id,
                response,
            } => {
                let removed =
                    self.remove_pending_outbound_response(&peer, connection_id, request_id);
                debug_assert!(
                    removed,
                    "Expect request_id to be pending before receiving response.",
                );

                let message = Message::Response {
                    request_id,
                    response,
                };
                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::Message {
                        peer,
                        connection_id,
                        message,
                    }));
            }
            handler::Event::Request {
                request_id,
                request,
                sender,
            } => match self.get_connection_mut(&peer, connection_id) {
                Some(connection) => {
                    let inserted = connection.pending_inbound_responses.insert(request_id);
                    debug_assert!(inserted, "Expect id of new request to be unknown.");

                    let channel = ResponseChannel { sender };
                    let message = Message::Request {
                        request_id,
                        request,
                        channel,
                    };
                    self.pending_events
                        .push_back(ToSwarm::GenerateEvent(Event::Message {
                            peer,
                            connection_id,
                            message,
                        }));
                }
                None => {
                    tracing::debug!("Connection ({connection_id}) closed after `Event::Request` ({request_id}) has been emitted.");
                }
            },
            handler::Event::ResponseSent(request_id) => {
                let removed =
                    self.remove_pending_inbound_response(&peer, connection_id, request_id);
                debug_assert!(
                    removed,
                    "Expect request_id to be pending before response is sent."
                );

                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::ResponseSent {
                        peer,
                        connection_id,
                        request_id,
                    }));
            }
            handler::Event::ResponseOmission(request_id) => {
                let removed =
                    self.remove_pending_inbound_response(&peer, connection_id, request_id);
                debug_assert!(
                    removed,
                    "Expect request_id to be pending before response is omitted.",
                );

                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::InboundFailure {
                        peer,
                        connection_id,
                        request_id,
                        error: InboundFailure::ResponseOmission,
                    }));
            }
            handler::Event::OutboundTimeout(request_id) => {
                let removed =
                    self.remove_pending_outbound_response(&peer, connection_id, request_id);
                debug_assert!(
                    removed,
                    "Expect request_id to be pending before request times out."
                );

                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::OutboundFailure {
                        peer,
                        connection_id,
                        request_id,
                        error: OutboundFailure::Timeout,
                    }));
            }
            handler::Event::OutboundUnsupportedProtocols(request_id) => {
                let removed =
                    self.remove_pending_outbound_response(&peer, connection_id, request_id);
                debug_assert!(
                    removed,
                    "Expect request_id to be pending before failing to connect.",
                );

                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::OutboundFailure {
                        peer,
                        connection_id,
                        request_id,
                        error: OutboundFailure::UnsupportedProtocols,
                    }));
            }
            handler::Event::OutboundStreamFailed { request_id, error } => {
                let removed =
                    self.remove_pending_outbound_response(&peer, connection_id, request_id);
                debug_assert!(removed, "Expect request_id to be pending upon failure");

                self.pending_events
                    .push_back(ToSwarm::GenerateEvent(Event::OutboundFailure {
                        peer,
                        connection_id,
                        request_id,
                        error: OutboundFailure::Io(error),
                    }))
            }
            handler::Event::InboundTimeout(request_id) => {
                let removed =
                    self.remove_pending_inbound_response(&peer, connection_id, request_id);

                if removed {
                    self.pending_events
                        .push_back(ToSwarm::GenerateEvent(Event::InboundFailure {
                            peer,
                            connection_id,
                            request_id,
                            error: InboundFailure::Timeout,
                        }));
                } else {
                    // This happens when timeout is emitted before `read_request` finishes.
                    tracing::debug!(
                        "Inbound request timeout for an unknown request_id ({request_id})"
                    );
                }
            }
            handler::Event::InboundStreamFailed { request_id, error } => {
                let removed =
                    self.remove_pending_inbound_response(&peer, connection_id, request_id);

                if removed {
                    self.pending_events
                        .push_back(ToSwarm::GenerateEvent(Event::InboundFailure {
                            peer,
                            connection_id,
                            request_id,
                            error: InboundFailure::Io(error),
                        }));
                } else {
                    // This happens when `read_request` fails.
                    tracing::debug!("Inbound failure is reported for an unknown request_id ({request_id}): {error}");
                }
            }
        }
    }

    #[tracing::instrument(level = "trace", name = "NetworkBehaviour::poll", skip(self))]
    fn poll(&mut self, _: &mut Context<'_>) -> Poll<ToSwarm<Self::ToSwarm, THandlerInEvent<Self>>> {
        if let Some(ev) = self.pending_events.pop_front() {
            return Poll::Ready(ev);
        } else if self.pending_events.capacity() > EMPTY_QUEUE_SHRINK_THRESHOLD {
            self.pending_events.shrink_to_fit();
        }

        Poll::Pending
    }
}

/// Internal threshold for when to shrink the capacity
/// of empty queues. If the capacity of an empty queue
/// exceeds this threshold, the associated memory is
/// released.
const EMPTY_QUEUE_SHRINK_THRESHOLD: usize = 100;

/// Internal information tracked for an established connection.
struct Connection {
    id: ConnectionId,
    remote_address: Option<Multiaddr>,
    /// Pending outbound responses where corresponding inbound requests have
    /// been received on this connection and emitted via `poll` but have not yet
    /// been answered.
    pending_outbound_responses: HashSet<OutboundRequestId>,
    /// Pending inbound responses for previously sent requests on this
    /// connection.
    pending_inbound_responses: HashSet<InboundRequestId>,
}

/// SwarmLLM patch: whether a connection is carried over a relay circuit.
///
/// Two signatures, because a relayed connection looks different depending on
/// which end dialled:
///
///  1. **We dialled through a relay** — the address carries an explicit
///     `/p2p-circuit` hop, e.g.
///     `/dns4/relay/tcp/8810/p2p/<relay>/p2p-circuit/p2p/<target>`.
///  2. **The peer dialled us through a relay** — the address is just
///     `/p2p/<peer>` with NO transport component at all. There is no `ip4`,
///     `udp` or `tcp` hop because there is no direct socket to describe.
///
/// Case 2 was missed originally, and the consequence was severe: selection below
/// picks the newest *direct* connection, and a relayed inbound connection is
/// typically the newest of the three a peer accumulates. So every request to
/// that peer went over a relay circuit that had already failed to negotiate,
/// producing a silent drop — no response, no `OutboundFailure` — until the
/// ACK-timeout sweep fired 10s later. Reproduced live 2026-07-26 between two
/// nodes on the same LAN that had a perfectly good direct QUIC connection
/// sitting unused.
///
/// When the address is unknown (`None`) we still treat the connection as direct.
/// That is the pre-existing default, and it is the safe direction: wrongly
/// calling a direct connection relayed would exclude it from selection.
fn connection_is_relayed(conn: &Connection) -> bool {
    conn.remote_address
        .as_ref()
        .is_some_and(|addr| !multiaddr_is_direct_transport(addr))
}

/// Does this multiaddr describe a real, direct transport path?
///
/// True only when it has a concrete network hop and no relay circuit. An address
/// consisting solely of `/p2p/<peer>` describes no path at all, which is what a
/// relay-carried inbound connection looks like.
fn multiaddr_is_direct_transport(addr: &Multiaddr) -> bool {
    use libp2p_core::multiaddr::Protocol;
    let mut has_transport = false;
    for p in addr.iter() {
        match p {
            Protocol::P2pCircuit => return false,
            Protocol::Ip4(_)
            | Protocol::Ip6(_)
            | Protocol::Dns(_)
            | Protocol::Dns4(_)
            | Protocol::Dns6(_)
            | Protocol::Dnsaddr(_)
            | Protocol::Tcp(_)
            | Protocol::Udp(_) => has_transport = true,
            _ => {}
        }
    }
    has_transport
}

impl Connection {
    fn new(id: ConnectionId, remote_address: Option<Multiaddr>) -> Self {
        Self {
            id,
            remote_address,
            pending_outbound_responses: Default::default(),
            pending_inbound_responses: Default::default(),
        }
    }
}


#[cfg(test)]
mod swarmllm_relay_selection_tests {
    use super::*;
    use std::str::FromStr;

    fn direct(addr: &str) -> bool {
        multiaddr_is_direct_transport(&Multiaddr::from_str(addr).unwrap())
    }

    /// Getting this wrong is expensive in BOTH directions: calling a relayed
    /// connection direct sends every request over a circuit that may be dead
    /// (silent drop, no OutboundFailure, 10s ACK timeout); calling a direct
    /// connection relayed excludes a good path from selection.
    #[test]
    fn direct_transports_are_recognised() {
        assert!(direct("/ip4/192.168.1.60/udp/8800/quic-v1"));
        assert!(direct(
            "/ip4/192.168.1.60/udp/8800/quic-v1/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY"
        ));
        assert!(direct("/ip4/212.132.104.177/tcp/8810"));
        assert!(direct("/dns4/swarmllm.duckdns.org/tcp/8810"));
        assert!(direct("/ip6/::1/udp/8800/quic-v1"));
    }

    #[test]
    fn explicit_circuit_hops_are_relayed() {
        assert!(!direct(
            "/dns4/swarmllm.duckdns.org/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G/p2p-circuit/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY"
        ));
        assert!(!direct(
            "/ip4/212.132.104.177/udp/8800/quic-v1/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G/p2p-circuit"
        ));
    }

    /// The case originally missed. A peer that dialled us through a relay shows
    /// up with no transport component at all — there is no socket to describe.
    /// Observed live 2026-07-26 as connection_id=117 to a LAN peer, selected in
    /// preference to two working direct QUIC connections because it was newest.
    #[test]
    fn a_bare_peer_address_is_relayed_not_direct() {
        assert!(!direct(
            "/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY"
        ));
    }

    #[test]
    fn an_unknown_address_stays_direct() {
        // `None` keeps the pre-existing default: excluding a possibly-good
        // connection is the worse error.
        let conn = Connection::new(ConnectionId::new_unchecked(1), None);
        assert!(!connection_is_relayed(&conn));
    }

    #[test]
    fn selection_prefers_a_direct_connection_over_a_newer_relayed_one() {
        // Mirrors the live inventory: two direct QUIC connections, then a
        // relay-carried inbound one established last.
        let conns = vec![
            Connection::new(
                ConnectionId::new_unchecked(1),
                Some(Multiaddr::from_str("/ip4/192.168.1.60/udp/8800/quic-v1").unwrap()),
            ),
            Connection::new(
                ConnectionId::new_unchecked(2),
                Some(
                    Multiaddr::from_str(
                        "/ip4/192.168.1.60/udp/8800/quic-v1/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY",
                    )
                    .unwrap(),
                ),
            ),
            Connection::new(
                ConnectionId::new_unchecked(117),
                Some(
                    Multiaddr::from_str(
                        "/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY",
                    )
                    .unwrap(),
                ),
            ),
        ];
        let newest_direct = conns.iter().rposition(|c| !connection_is_relayed(c));
        let ix = newest_direct.unwrap_or(conns.len() - 1);
        assert_eq!(
            conns[ix].id,
            ConnectionId::new_unchecked(2),
            "must pick the newest DIRECT connection, not the newer relayed one"
        );
    }

    fn pick(conns: &[Connection]) -> ConnectionId {
        let best_direct = conns
            .iter()
            .enumerate()
            .filter(|(_, c)| !connection_is_relayed(c))
            .min_by_key(|(i, c)| (c.pending_outbound_responses.len(), std::cmp::Reverse(*i)))
            .map(|(i, _)| i);
        conns[best_direct.unwrap_or(conns.len() - 1)].id
    }

    fn direct_conn(id: u64, addr: &str, pending: &[u64]) -> Connection {
        let mut c = Connection::new(
            ConnectionId::new_unchecked(id as usize),
            Some(Multiaddr::from_str(addr).unwrap()),
        );
        for r in pending {
            c.pending_outbound_responses
                .insert(OutboundRequestId(*r as u64));
        }
        c
    }

    const A: &str = "/ip4/192.168.1.60/udp/8800/quic-v1";
    const B: &str = "/ip4/192.168.1.60/udp/8800/quic-v1/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY";

    /// Age alone picked a newer connection that silently swallowed every send
    /// while an older, demonstrably working one sat unused. Un-answered
    /// requests are the signal that distinguishes them.
    #[test]
    fn a_direct_connection_that_is_answering_beats_a_newer_silent_one() {
        let conns = vec![
            direct_conn(1, A, &[]),           // older, answering
            direct_conn(2, B, &[10, 11, 12]), // newer, nothing coming back
        ];
        assert_eq!(pick(&conns), ConnectionId::new_unchecked(1));
    }

    #[test]
    fn among_equally_idle_connections_the_newest_still_wins() {
        // Preserves the DCUtR behaviour: an upgraded direct connection is the
        // newest and must win when nothing distinguishes them.
        let conns = vec![direct_conn(1, A, &[]), direct_conn(2, B, &[])];
        assert_eq!(pick(&conns), ConnectionId::new_unchecked(2));
    }

    #[test]
    fn a_busy_but_healthy_connection_is_not_abandoned_for_a_relay() {
        // Relayed is excluded regardless of how many requests are in flight on
        // the direct path — a circuit is the worse route even when busy.
        let mut relayed = Connection::new(
            ConnectionId::new_unchecked(3),
            Some(Multiaddr::from_str("/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY").unwrap()),
        );
        relayed.pending_outbound_responses.clear();
        let conns = vec![direct_conn(1, A, &[10, 11]), relayed];
        assert_eq!(pick(&conns), ConnectionId::new_unchecked(1));
    }

    #[test]
    fn falls_back_to_a_relayed_connection_when_that_is_all_there_is() {
        // A NAT'd holder reachable only via relay must stay usable.
        let conns = vec![Connection::new(
            ConnectionId::new_unchecked(9),
            Some(Multiaddr::from_str("/p2p/12D3KooWKwvCNmumN89DftJbEC1yRcnP1YxVFKEXMLCo7EzifsaY").unwrap()),
        )];
        let newest_direct = conns.iter().rposition(|c| !connection_is_relayed(c));
        let ix = newest_direct.unwrap_or(conns.len() - 1);
        assert_eq!(conns[ix].id, ConnectionId::new_unchecked(9));
    }
}
