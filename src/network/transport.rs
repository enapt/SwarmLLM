use libp2p::identity::Keypair;
use libp2p::PeerId;

use crate::error::SwarmError;
use crate::types::NodeId;

/// Convert an Ed25519 signing key (32 bytes) to a libp2p Keypair.
pub fn ed25519_to_libp2p_keypair(signing_key_bytes: [u8; 32]) -> Result<Keypair, SwarmError> {
    let mut key_bytes = signing_key_bytes;
    Keypair::ed25519_from_bytes(&mut key_bytes)
        .map_err(|e| SwarmError::Network(format!("Failed to create libp2p keypair: {e}")))
}

/// Convert a NodeId (Ed25519 public key bytes) to a libp2p PeerId.
///
/// Both NodeId and PeerId derive from the same Ed25519 key, so the conversion
/// is deterministic. Returns None if the bytes are not a valid Ed25519 public key.
///
/// Was test-only until the diagnostics endpoint needed the local PeerId to
/// filter the peer cache the same way the network manager does — that filter
/// drops any address routing through our own id, so it needs the id.
pub fn node_id_to_peer_id(node_id: &NodeId) -> Option<PeerId> {
    let ed_pk = libp2p::identity::ed25519::PublicKey::try_from_bytes(&node_id.0).ok()?;
    let pk = libp2p::identity::PublicKey::from(ed_pk);
    Some(pk.to_peer_id())
}

/// Extract NodeId (Ed25519 public key bytes) from a libp2p PeerId.
///
/// This only works for Ed25519-based PeerIds where the public key is inlined
/// in the multihash (identity hash). Returns None for other PeerId types.
pub fn peer_id_to_node_id(peer_id: &PeerId) -> Option<NodeId> {
    // PeerId for Ed25519 keys uses identity multihash, so the public key
    // can be extracted directly from the PeerId bytes.
    // .get(2..) — return None on a malformed/short PeerId rather than
    // panicking on an unchecked slice (R96).
    let bytes = peer_id.to_bytes();
    let pk = libp2p::identity::PublicKey::try_decode_protobuf(bytes.get(2..)?).ok()?;
    let ed_pk = pk.try_into_ed25519().ok()?;
    let bytes = ed_pk.to_bytes();
    Some(NodeId(bytes))
}

/// The computer behind a libp2p PeerId carried as bytes — the shape the
/// dispatcher is handed for an inbound message's authenticated sender.
pub fn node_id_of_peer_bytes(peer_bytes: &[u8]) -> Option<NodeId> {
    peer_id_to_node_id(&PeerId::from_bytes(peer_bytes).ok()?)
}

/// The congestion controller this node's QUIC connections run: quinn's BBR,
/// which paces a send by the delivery rate it has measured. quinn's default,
/// Cubic, paces every send at window / round trip, and grows the window only
/// while a send fills it — a split's traffic is separate bursts (one a hop a
/// pass) that never do, so each burst was spread over ~0.4 of a round trip, and
/// a loss shrank the window for good (`docs/invariants/network.md` § "A burst
/// is paced by the link, not by a window it never fills"). Sender-side only: a
/// peer running either still talks to us. `SWARMLLM_QUIC_CC=cubic` is the A/B.
pub fn quic_congestion() -> libp2p::quic::Congestion {
    congestion_switch(std::env::var("SWARMLLM_QUIC_CC").ok().as_deref())
}

/// The switch's reading, apart from the environment so a test can ask it.
fn congestion_switch(v: Option<&str>) -> libp2p::quic::Congestion {
    match v {
        Some("cubic") => libp2p::quic::Congestion::Cubic,
        _ => libp2p::quic::Congestion::Bbr,
    }
}

/// What a dial offline mode stopped fails with. Said as an error, never as
/// "address not supported": an `or_transport` passes an unsupported address on
/// to the next transport, and the log would read as a missing protocol.
#[derive(Debug, thiserror::Error)]
#[error("offline mode: this computer connects only to computers on its own network")]
pub struct OfflineRefusal;

/// Offline mode's one gate (#468): every connection this node dials goes
/// through it, and while `pool.offline_mode` is on ("air-gapped: no internet,
/// mDNS only") it refuses any address on the open internet BEFORE a byte is
/// sent — whichever behaviour asked for the dial: peer exchange, Kademlia's own
/// walks, a redial, a chain's next hop, a relay hop. Each of those once checked
/// the switch for itself, and the ones that did not dialled the internet as
/// soon as a computer on the same network named a public address (#468).
///
/// "The open internet" is `pool::invite::multiaddr_is_internet_reachable` —
/// the predicate AutoNAT and the lost-peer lookup already share: a public IP or
/// a DNS name. Private ranges, loopback, link-local, CGNAT (Tailscale) and IPv6
/// unique-local stay allowed. A name is refused rather than resolved, since the
/// lookup itself leaves the machine; mDNS names a peer by its address.
///
/// It sits OUTSIDE the name resolver (so it sees the address as written) and
/// INSIDE the relay client: a circuit dial becomes a dial of its relay hop,
/// which comes back through here. A connection ARRIVING from such an address
/// (a port the router still forwards, a peer that knew our address from before)
/// is dropped as the transport hands it up, before its security handshake —
/// after TCP's own handshake, or one QUIC round trip. Connections already open
/// when the switch is turned on stay open. The switch is read on every dial and
/// every arrival, so turning it on or off applies to the next one.
///
/// Prior art: go-libp2p's `ConnectionGater.InterceptAddrDial(peer, addr)`,
/// asked for every address before it is dialled, and `InterceptAccept`, asked
/// when a listener receives a connection, before any upgrade (in go-libp2p too
/// after TCP's handshake, and one round trip for QUIC). rust-libp2p has no gater, and
/// a behaviour cannot be one: `handle_pending_outbound_connection` sees only a
/// dial's own addresses (Kademlia and Identify add theirs after, libp2p-swarm
/// 0.47.1 `lib.rs:458-469`), and `handle_established_outbound_connection` runs
/// after TCP, noise and yamux have already talked to the far end.
pub struct OfflineGate<O> {
    inner: libp2p::core::transport::Boxed<O>,
    offline: Box<dyn Fn() -> bool + Send + Sync>,
}

impl<O> OfflineGate<O> {
    pub fn new(
        inner: libp2p::core::transport::Boxed<O>,
        offline: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            inner,
            offline: Box::new(offline),
        }
    }

    /// Whether a dial of `addr` is refused now.
    fn refuses(&self, addr: &libp2p::Multiaddr) -> bool {
        (self.offline)() && crate::pool::invite::multiaddr_is_internet_reachable(addr)
    }
}

impl<O> libp2p::core::Transport for OfflineGate<O> {
    type Output = O;
    type Error = std::io::Error;
    type ListenerUpgrade =
        <libp2p::core::transport::Boxed<O> as libp2p::core::Transport>::ListenerUpgrade;
    type Dial = <libp2p::core::transport::Boxed<O> as libp2p::core::Transport>::Dial;

    fn listen_on(
        &mut self,
        id: libp2p::core::transport::ListenerId,
        addr: libp2p::Multiaddr,
    ) -> Result<(), libp2p::core::transport::TransportError<Self::Error>> {
        self.inner.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: libp2p::core::transport::ListenerId) -> bool {
        self.inner.remove_listener(id)
    }

    fn dial(
        &mut self,
        addr: libp2p::Multiaddr,
        opts: libp2p::core::transport::DialOpts,
    ) -> Result<Self::Dial, libp2p::core::transport::TransportError<Self::Error>> {
        if self.refuses(&addr) {
            tracing::debug!(
                target: "swarmllm::network::transport",
                %addr,
                "Offline mode: refused to dial an address off this network (#468)"
            );
            return Err(libp2p::core::transport::TransportError::Other(
                std::io::Error::new(std::io::ErrorKind::PermissionDenied, OfflineRefusal),
            ));
        }
        self.inner.dial(addr, opts)
    }

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<libp2p::core::transport::TransportEvent<Self::ListenerUpgrade, Self::Error>>
    {
        loop {
            match std::pin::Pin::new(&mut self.inner).poll(cx) {
                std::task::Poll::Ready(libp2p::core::transport::TransportEvent::Incoming {
                    send_back_addr,
                    ..
                }) if self.refuses(&send_back_addr) => {
                    // Dropping the upgrade closes the connection before noise.
                    tracing::debug!(
                        target: "swarmllm::network::transport",
                        %send_back_addr,
                        "Offline mode: refused a connection from an address off this network (#468)"
                    );
                }
                other => return other,
            }
        }
    }
}

/// The transport every swarm of this node is built on: TCP exactly as
/// libp2p 0.56's `SwarmBuilder::with_tcp` builds it (`builder/phase/tcp.rs`:
/// multistream-select `V1Lazy`, noise, then yamux — here through the
/// per-protocol meter), QUIC with this node's congestion controller and its
/// connection metered (it has no multiplexer argument), names resolved by the
/// system resolver as `with_dns` does, and offline mode's gate outside it all.
/// Built by hand because the builder's own TCP and DNS phases wrap whatever
/// came before them, which would put the resolver outside the gate. The node
/// (`network::manager`) and `protocol_traffic`'s known-answer test both build
/// through here, so the test measures the stack the node runs.
pub fn swarm_transport(
    key: &Keypair,
    by_protocol: std::sync::Arc<crate::network::protocol_traffic::ProtocolTrafficMeter>,
    offline: impl Fn() -> bool + Send + Sync + 'static,
) -> Result<
    OfflineGate<(PeerId, libp2p::core::muxing::StreamMuxerBox)>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    use libp2p::core::muxing::StreamMuxerBox;
    use libp2p::core::Transport as _;
    let tcp = libp2p::tcp::tokio::Transport::new(libp2p::tcp::Config::default().nodelay(true))
        .upgrade(libp2p::core::upgrade::Version::V1Lazy)
        .authenticate(libp2p::noise::Config::new(key)?)
        // Use yamux 0.13 defaults (auto-tuned windows, 1 GiB max connection window).
        // NOTE: Do NOT call set_receive_window_size or set_max_buffer_size — those
        // are deprecated and silently downgrade to yamux 0.12 which has severe
        // substream opening delays (~30s between successful outbound requests).
        .multiplex(crate::network::protocol_traffic::metered_muxer(
            libp2p::yamux::Config::default,
            by_protocol.clone(),
        )())
        .map(|(peer, muxer), _| (peer, StreamMuxerBox::new(muxer)));
    let quic = libp2p::quic::tokio::Transport::new(
        libp2p::quic::Config::new(key).congestion(quic_congestion()),
    )
    .map(move |(peer, connection), _| {
        (
            peer,
            StreamMuxerBox::new(crate::network::protocol_traffic::MeteredMuxer::new(
                connection,
                by_protocol.clone(),
            )),
        )
    });
    let either = tcp.or_transport(quic).map(|either, _| either.into_inner());
    // `/dns4` / `/dns6` / `/dnsaddr` would otherwise fail "Multiaddr is not
    // supported" — the default bootstrap anchor is a DNS name.
    let resolved = libp2p::dns::tokio::Transport::system(either)?;
    Ok(OfflineGate::new(resolved.boxed(), offline))
}

#[cfg(test)]
mod offline_gate_tests {
    use super::*;
    use libp2p::core::transport::{DialOpts, ListenerId, PortUse, TransportError, TransportEvent};
    use libp2p::core::{Endpoint, Transport};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A transport that counts the dials that reach it and connects nothing.
    struct CountsDials(Arc<AtomicUsize>);

    impl Transport for CountsDials {
        type Output = ();
        type Error = std::io::Error;
        type ListenerUpgrade = futures::future::Pending<Result<(), std::io::Error>>;
        type Dial = futures::future::Pending<Result<(), std::io::Error>>;

        fn listen_on(
            &mut self,
            _: ListenerId,
            addr: libp2p::Multiaddr,
        ) -> Result<(), TransportError<Self::Error>> {
            Err(TransportError::MultiaddrNotSupported(addr))
        }
        fn remove_listener(&mut self, _: ListenerId) -> bool {
            false
        }
        fn dial(
            &mut self,
            _: libp2p::Multiaddr,
            _: DialOpts,
        ) -> Result<Self::Dial, TransportError<Self::Error>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(futures::future::pending())
        }
        fn poll(
            self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
            std::task::Poll::Pending
        }
    }

    /// A listener that hands up one connection from each address queued.
    struct Arrivals(std::collections::VecDeque<libp2p::Multiaddr>);

    impl Transport for Arrivals {
        type Output = ();
        type Error = std::io::Error;
        type ListenerUpgrade = futures::future::Pending<Result<(), std::io::Error>>;
        type Dial = futures::future::Pending<Result<(), std::io::Error>>;

        fn listen_on(
            &mut self,
            _: ListenerId,
            addr: libp2p::Multiaddr,
        ) -> Result<(), TransportError<Self::Error>> {
            Err(TransportError::MultiaddrNotSupported(addr))
        }
        fn remove_listener(&mut self, _: ListenerId) -> bool {
            false
        }
        fn dial(
            &mut self,
            addr: libp2p::Multiaddr,
            _: DialOpts,
        ) -> Result<Self::Dial, TransportError<Self::Error>> {
            Err(TransportError::MultiaddrNotSupported(addr))
        }
        fn poll(
            mut self: std::pin::Pin<&mut Self>,
            _: &mut std::task::Context<'_>,
        ) -> std::task::Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
            match self.0.pop_front() {
                Some(send_back_addr) => std::task::Poll::Ready(TransportEvent::Incoming {
                    listener_id: ListenerId::next(),
                    upgrade: futures::future::pending(),
                    local_addr: "/ip4/192.168.1.53/tcp/8810".parse().unwrap(),
                    send_back_addr,
                }),
                None => std::task::Poll::Pending,
            }
        }
    }

    /// Every connection the listener hands up, in order, until it has none.
    fn arrivals_seen(offline: bool, from: &[&str]) -> Vec<String> {
        let mut gate = OfflineGate::new(
            Arrivals(from.iter().map(|a| a.parse().unwrap()).collect()).boxed(),
            move || offline,
        );
        let mut cx = std::task::Context::from_waker(futures::task::noop_waker_ref());
        let mut seen = Vec::new();
        while let std::task::Poll::Ready(event) = std::pin::Pin::new(&mut gate).poll(&mut cx) {
            if let TransportEvent::Incoming { send_back_addr, .. } = event {
                seen.push(send_back_addr.to_string());
            }
        }
        seen
    }

    /// A port the router still forwards, or a peer that knew this computer's
    /// public address from before offline mode, reaches the listener: offline,
    /// such a connection is dropped before its handshake, and one from this
    /// network still comes through.
    #[test]
    fn offline_mode_drops_a_connection_arriving_from_the_open_internet() {
        let from = [
            "/ip4/95.233.105.157/tcp/51000",
            "/ip4/192.168.1.200/tcp/51001",
        ];
        assert_eq!(
            arrivals_seen(true, &from),
            ["/ip4/192.168.1.200/tcp/51001"],
            "only the computer on this network is let in"
        );
        assert_eq!(arrivals_seen(false, &from).len(), 2, "online lets both in");
    }

    fn gate(offline: Arc<AtomicBool>) -> (OfflineGate<()>, Arc<AtomicUsize>) {
        let dials = Arc::new(AtomicUsize::new(0));
        let gate = OfflineGate::new(CountsDials(dials.clone()).boxed(), move || {
            offline.load(Ordering::SeqCst)
        });
        (gate, dials)
    }

    fn opts() -> DialOpts {
        DialOpts {
            role: Endpoint::Dialer,
            port_use: PortUse::New,
        }
    }

    fn refused(result: Result<impl Sized, TransportError<std::io::Error>>) -> bool {
        match result {
            Err(TransportError::Other(e)) => e
                .get_ref()
                .is_some_and(|inner| inner.is::<OfflineRefusal>()),
            _ => false,
        }
    }

    const OPEN_INTERNET: [&str; 5] = [
        "/ip4/95.233.105.157/tcp/8810",
        "/ip4/212.132.104.177/udp/8800/quic-v1",
        "/ip6/2606:4700:4700::1111/tcp/8810",
        "/dns4/swarmllm.duckdns.org/tcp/8810",
        "/dnsaddr/bootstrap.example.org",
    ];

    const THIS_NETWORK: [&str; 7] = [
        "/ip4/192.168.1.20/tcp/8810",
        "/ip4/10.0.0.7/udp/8800/quic-v1",
        "/ip4/127.0.0.1/tcp/8810",
        "/ip4/100.101.2.3/tcp/8810",
        "/ip4/169.254.3.4/tcp/8810",
        "/ip6/fd12:3456::1/tcp/8810",
        "/ip6/fe80::1/tcp/8810",
    ];

    /// The probe of 2026-11-16 saw an offline computer dial the anchor and a
    /// peer in Italy within 5 s of a LAN peer naming them. Nothing on the open
    /// internet reaches the transport underneath; every address on this
    /// network still does, Tailscale's CGNAT range included.
    #[test]
    fn offline_mode_dials_nothing_on_the_open_internet() {
        let (mut gate, dials) = gate(Arc::new(AtomicBool::new(true)));
        for addr in OPEN_INTERNET {
            assert!(
                refused(gate.dial(addr.parse().unwrap(), opts())),
                "{addr} must be refused offline"
            );
        }
        assert_eq!(dials.load(Ordering::SeqCst), 0, "nothing reached the wire");
        for addr in THIS_NETWORK {
            assert!(
                gate.dial(addr.parse().unwrap(), opts()).is_ok(),
                "{addr} is on this network"
            );
        }
        assert_eq!(dials.load(Ordering::SeqCst), THIS_NETWORK.len());
    }

    /// The switch is the live setting, read at each dial: turning offline mode
    /// on or off from the dashboard applies to the next dial, no restart.
    #[test]
    fn the_switch_is_read_at_every_dial() {
        let offline = Arc::new(AtomicBool::new(false));
        let (mut gate, dials) = gate(offline.clone());
        let public: libp2p::Multiaddr = OPEN_INTERNET[0].parse().unwrap();
        assert!(
            gate.dial(public.clone(), opts()).is_ok(),
            "online dials anything"
        );
        offline.store(true, Ordering::SeqCst);
        assert!(refused(gate.dial(public.clone(), opts())));
        offline.store(false, Ordering::SeqCst);
        assert!(gate.dial(public, opts()).is_ok());
        assert_eq!(dials.load(Ordering::SeqCst), 2);
    }

    fn swarm(offline: bool) -> libp2p::Swarm<libp2p::swarm::dummy::Behaviour> {
        libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_other_transport(|key| {
                swarm_transport(
                    key,
                    Arc::new(crate::network::protocol_traffic::ProtocolTrafficMeter::default()),
                    move || offline,
                )
            })
            .expect("transport")
            .with_behaviour(|_| libp2p::swarm::dummy::Behaviour)
            .expect("behaviour")
            .with_swarm_config(|c| {
                c.with_idle_connection_timeout(std::time::Duration::from_secs(10))
            })
            .build()
    }

    /// The node's own stack, offline: a computer on this network is reached
    /// over TCP and over QUIC, and a public address fails at once with the
    /// refusal — from the swarm, through every layer the builder adds.
    #[tokio::test]
    async fn an_offline_node_reaches_this_network_and_refuses_the_internet() {
        use futures::StreamExt;
        use libp2p::swarm::SwarmEvent;
        for listen in ["/ip4/127.0.0.1/tcp/0", "/ip4/127.0.0.1/udp/0/quic-v1"] {
            let mut far = swarm(false);
            far.listen_on(listen.parse().unwrap()).unwrap();
            let far_addr = loop {
                if let SwarmEvent::NewListenAddr { address, .. } = far.select_next_some().await {
                    break address;
                }
            };
            let mut near = swarm(true);
            near.dial(far_addr.clone()).unwrap();
            let reached = tokio::time::timeout(std::time::Duration::from_secs(10), async {
                loop {
                    tokio::select! {
                        e = near.select_next_some() => match e {
                            SwarmEvent::ConnectionEstablished { .. } => return Ok(()),
                            SwarmEvent::OutgoingConnectionError { error, .. } => {
                                return Err(error.to_string())
                            }
                            _ => {}
                        },
                        _ = far.select_next_some() => {}
                    }
                }
            })
            .await
            .expect("the dial ended");
            assert_eq!(reached, Ok(()), "{far_addr} is on this computer");

            let started = std::time::Instant::now();
            near.dial(
                "/ip4/203.0.113.7/tcp/9"
                    .parse::<libp2p::Multiaddr>()
                    .unwrap(),
            )
            .unwrap();
            let error = tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    if let SwarmEvent::OutgoingConnectionError { error, .. } =
                        near.select_next_some().await
                    {
                        // `Debug`: libp2p's `TransportError::Other` prints
                        // nothing of its own, its source chain is the message.
                        return format!("{error:?}");
                    }
                }
            })
            .await
            .expect("refused at once, not timed out");
            assert!(error.contains("OfflineRefusal"), "{error}");
            assert!(started.elapsed() < std::time::Duration::from_secs(2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quic_connections_run_bbr_unless_switched_back_to_cubic() {
        assert_eq!(congestion_switch(None), libp2p::quic::Congestion::Bbr);
        assert_eq!(
            congestion_switch(Some("bbr")),
            libp2p::quic::Congestion::Bbr
        );
        assert_eq!(
            congestion_switch(Some("cubic")),
            libp2p::quic::Congestion::Cubic
        );
    }

    #[test]
    fn keypair_conversion() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let result = ed25519_to_libp2p_keypair(signing_key.to_bytes());
        assert!(result.is_ok());

        let keypair = result.unwrap();
        assert!(keypair
            .public()
            .to_peer_id()
            .to_string()
            .starts_with("12D3"));
    }

    #[test]
    fn node_id_peer_id_roundtrip() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = NodeId(signing_key.verifying_key().to_bytes());

        // NodeId → PeerId
        let peer_id = node_id_to_peer_id(&node_id).expect("should convert");

        // PeerId → NodeId
        let recovered = peer_id_to_node_id(&peer_id).expect("should recover");
        assert_eq!(node_id, recovered);
    }

    #[test]
    fn node_id_to_peer_id_matches_keypair() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let signing_key = SigningKey::generate(&mut OsRng);
        let node_id = NodeId(signing_key.verifying_key().to_bytes());

        // Convert via keypair (existing path)
        let keypair = ed25519_to_libp2p_keypair(signing_key.to_bytes()).unwrap();
        let peer_id_from_keypair = keypair.public().to_peer_id();

        // Convert via node_id (new path)
        let peer_id_from_node_id = node_id_to_peer_id(&node_id).unwrap();

        assert_eq!(peer_id_from_keypair, peer_id_from_node_id);
    }
}
