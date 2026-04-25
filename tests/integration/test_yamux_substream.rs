//! Minimal reproduction test for yamux substream stalling.
//!
//! Tests sequential request_response exchanges over different transports
//! to isolate whether the 2nd-substream stall is yamux-specific, WSL2-specific,
//! or related to libp2p polling.
//!
//! Run with: cargo test --test yamux_substream -- --nocapture --test-threads=1

use std::time::{Duration, Instant};

use futures::StreamExt;
use libp2p::request_response::{self, ProtocolSupport};
use libp2p::swarm::{NetworkBehaviour, SwarmEvent};
use libp2p::{noise, yamux, Multiaddr, PeerId, StreamProtocol, Swarm, SwarmBuilder};
use tokio::time::timeout;

/// Simple codec that sends/receives raw bytes.
#[derive(Debug, Clone, Default)]
struct RawCodec;

#[derive(Debug, Clone, PartialEq)]
struct RawRequest(Vec<u8>);

#[derive(Debug, Clone, PartialEq)]
struct RawResponse(Vec<u8>);

#[async_trait::async_trait]
impl request_response::Codec for RawCodec {
    type Protocol = StreamProtocol;
    type Request = RawRequest;
    type Response = RawResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        futures::AsyncReadExt::read_exact(io, &mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        futures::AsyncReadExt::read_exact(io, &mut buf).await?;
        Ok(RawRequest(buf))
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        let mut len_buf = [0u8; 4];
        futures::AsyncReadExt::read_exact(io, &mut len_buf).await?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut buf = vec![0u8; len];
        futures::AsyncReadExt::read_exact(io, &mut buf).await?;
        Ok(RawResponse(buf))
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        let len = req.0.len() as u32;
        futures::AsyncWriteExt::write_all(io, &len.to_be_bytes()).await?;
        futures::AsyncWriteExt::write_all(io, &req.0).await?;
        Ok(())
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        resp: Self::Response,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        let len = resp.0.len() as u32;
        futures::AsyncWriteExt::write_all(io, &len.to_be_bytes()).await?;
        futures::AsyncWriteExt::write_all(io, &resp.0).await?;
        Ok(())
    }
}

#[derive(NetworkBehaviour)]
struct MinimalBehaviour {
    request_response: request_response::Behaviour<RawCodec>,
}

fn build_tcp_yamux_swarm() -> Swarm<MinimalBehaviour> {
    let protocol = StreamProtocol::new("/test/1.0.0");
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .unwrap()
        .with_behaviour(|_| {
            Ok(MinimalBehaviour {
                request_response: request_response::Behaviour::new(
                    [(protocol.clone(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(30)),
                ),
            })
        })
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build()
}

/// Compile/run guard for the yamux 0.13 deprecated-config path
/// (`set_receive_window_size` / `set_max_buffer_size`). The tests using this
/// helper assert correctness only — they do NOT measure that the deprecated
/// configuration degrades performance, since the documented degradation is
/// substream-opening latency that's hard to reproduce deterministically in
/// CI. Production code in `manager/mod.rs` explicitly forbids these calls
/// (see the comment near `with_yamux_config`); this test exists so a future
/// refactor that re-enables them shows up as a behaviour change rather than
/// a silent regression.
fn build_tcp_yamux_16mb_swarm() -> Swarm<MinimalBehaviour> {
    let protocol = StreamProtocol::new("/test/1.0.0");
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default().nodelay(true),
            noise::Config::new,
            #[allow(deprecated)]
            || {
                let mut cfg = yamux::Config::default();
                cfg.set_receive_window_size(16 * 1024 * 1024);
                cfg.set_max_buffer_size(16 * 1024 * 1024);
                cfg
            },
        )
        .unwrap()
        .with_behaviour(|_| {
            Ok(MinimalBehaviour {
                request_response: request_response::Behaviour::new(
                    [(protocol.clone(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(30)),
                ),
            })
        })
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build()
}

fn build_quic_swarm() -> Swarm<MinimalBehaviour> {
    let protocol = StreamProtocol::new("/test/1.0.0");
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_quic()
        .with_behaviour(|_| {
            Ok(MinimalBehaviour {
                request_response: request_response::Behaviour::new(
                    [(protocol.clone(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(30)),
                ),
            })
        })
        .unwrap()
        .with_swarm_config(|c| c.with_idle_connection_timeout(Duration::from_secs(60)))
        .build()
}

/// Run N sequential request_response round-trips between two swarms.
/// Returns the per-round-trip durations, or an error message.
async fn run_sequential_exchanges(
    swarm1: &mut Swarm<MinimalBehaviour>,
    swarm2: &mut Swarm<MinimalBehaviour>,
    peer2_id: PeerId,
    count: usize,
    payload_size: usize,
) -> Result<Vec<Duration>, String> {
    let mut durations = Vec::new();
    let payload = vec![0xABu8; payload_size];

    for i in 0..count {
        let start = Instant::now();
        let request = RawRequest(payload.clone());

        // Send request from swarm1 to swarm2
        let outbound_id = swarm1
            .behaviour_mut()
            .request_response
            .send_request(&peer2_id, request.clone());

        eprintln!(
            "  [round {}] sent request (outbound_id={:?}, payload={}B)",
            i + 1,
            outbound_id,
            payload_size
        );

        // Poll both swarms until we get the response
        let result = timeout(Duration::from_secs(15), async {
            let mut _request_served = false;

            loop {
                tokio::select! {
                    event = swarm1.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(MinimalBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    message: request_response::Message::Response { response, .. },
                                    ..
                                },
                            )) => {
                                assert_eq!(response.0.len(), payload_size);
                                break;
                            }
                            SwarmEvent::Behaviour(MinimalBehaviourEvent::RequestResponse(
                                request_response::Event::OutboundFailure { error, .. },
                            )) => {
                                return Err(format!("OutboundFailure on round {}: {:?}", i + 1, error));
                            }
                            _ => {}
                        }
                    }
                    event = swarm2.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(MinimalBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    peer,
                                    message: request_response::Message::Request { request, channel, .. },
                                    ..
                                },
                            )) => {
                                // Echo back same-size response
                                let resp = RawResponse(request.0);
                                if let Err(e) = swarm2.behaviour_mut().request_response.send_response(channel, resp) {
                                    return Err(format!("send_response failed on round {}: {:?}", i + 1, e));
                                }
                                _request_served = true;
                                eprintln!("  [round {}] swarm2 received request from {:?}, sent response", i + 1, &peer.to_string()[..8]);
                            }
                            SwarmEvent::Behaviour(MinimalBehaviourEvent::RequestResponse(
                                request_response::Event::InboundFailure { error, .. },
                            )) => {
                                eprintln!("  [round {}] InboundFailure: {:?}", i + 1, error);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Ok(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                let elapsed = start.elapsed();
                eprintln!("  [round {}] completed in {:?}", i + 1, elapsed);
                durations.push(elapsed);
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(format!("TIMEOUT on round {} (15s)", i + 1)),
        }
    }

    Ok(durations)
}

/// Connect swarm1 to swarm2 and wait for the connection to be established.
async fn connect_swarms(
    swarm1: &mut Swarm<MinimalBehaviour>,
    swarm2: &mut Swarm<MinimalBehaviour>,
) -> Result<(PeerId, PeerId), String> {
    let peer1_id = *swarm1.local_peer_id();
    let peer2_id = *swarm2.local_peer_id();

    // Listen on swarm2
    let listen_addr: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
    swarm2.listen_on(listen_addr).unwrap();

    // Wait for swarm2 to start listening
    let swarm2_addr = timeout(Duration::from_secs(5), async {
        loop {
            if let SwarmEvent::NewListenAddr { address, .. } = swarm2.select_next_some().await {
                if address.to_string().contains("127.0.0.1") {
                    return address;
                }
            }
        }
    })
    .await
    .map_err(|_| "Timeout waiting for swarm2 listen address")?;

    eprintln!(
        "  swarm2 listening on {} (peer {})",
        swarm2_addr,
        &peer2_id.to_string()[..8]
    );

    // Dial from swarm1 to swarm2
    swarm1
        .dial(swarm2_addr.clone())
        .map_err(|e| format!("Dial failed: {e}"))?;

    // Wait for connection established on both sides
    let connected = timeout(Duration::from_secs(10), async {
        let mut s1_connected = false;
        let mut s2_connected = false;
        loop {
            tokio::select! {
                event = swarm1.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer2_id {
                            s1_connected = true;
                            eprintln!("  swarm1 connected to swarm2");
                        }
                    }
                }
                event = swarm2.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer1_id {
                            s2_connected = true;
                            eprintln!("  swarm2 connected to swarm1");
                        }
                    }
                }
            }
            if s1_connected && s2_connected {
                break;
            }
        }
    })
    .await;

    connected.map_err(|_| "Timeout waiting for connection establishment")?;

    // Small delay to let the connection fully negotiate
    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok((peer1_id, peer2_id))
}

/// Helper to connect QUIC swarms (different listen addr format).
async fn connect_quic_swarms(
    swarm1: &mut Swarm<MinimalBehaviour>,
    swarm2: &mut Swarm<MinimalBehaviour>,
) -> Result<(PeerId, PeerId), String> {
    let peer1_id = *swarm1.local_peer_id();
    let peer2_id = *swarm2.local_peer_id();

    let listen_addr: Multiaddr = "/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap();
    swarm2.listen_on(listen_addr).unwrap();

    let swarm2_addr = timeout(Duration::from_secs(5), async {
        loop {
            if let SwarmEvent::NewListenAddr { address, .. } = swarm2.select_next_some().await {
                if address.to_string().contains("127.0.0.1") {
                    return address;
                }
            }
        }
    })
    .await
    .map_err(|_| "Timeout waiting for swarm2 QUIC listen address")?;

    eprintln!(
        "  swarm2 QUIC listening on {} (peer {})",
        swarm2_addr,
        &peer2_id.to_string()[..8]
    );

    swarm1
        .dial(swarm2_addr.clone())
        .map_err(|e| format!("QUIC dial failed: {e}"))?;

    let connected = timeout(Duration::from_secs(10), async {
        let mut s1_connected = false;
        let mut s2_connected = false;
        loop {
            tokio::select! {
                event = swarm1.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer2_id {
                            s1_connected = true;
                            eprintln!("  swarm1 QUIC connected to swarm2");
                        }
                    }
                }
                event = swarm2.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer1_id {
                            s2_connected = true;
                            eprintln!("  swarm2 QUIC connected to swarm1");
                        }
                    }
                }
            }
            if s1_connected && s2_connected {
                break;
            }
        }
    })
    .await;

    connected.map_err(|_| "Timeout waiting for QUIC connection establishment")?;

    tokio::time::sleep(Duration::from_millis(200)).await;

    Ok((peer1_id, peer2_id))
}

// ─── Multi-behaviour swarm (mimics SwarmBehaviour) ───

/// Behaviour with kademlia + gossipsub + identify + request_response
/// to test whether poll competition causes substream starvation.
#[derive(NetworkBehaviour)]
struct FullBehaviour {
    // request_response first (same as our production SwarmBehaviour)
    request_response: request_response::Behaviour<RawCodec>,
    kademlia: kad::Behaviour<kad::store::MemoryStore>,
    gossipsub: gossipsub::Behaviour,
    identify: identify::Behaviour,
}

use libp2p::{gossipsub, identify, kad};

fn build_full_behaviour_swarm() -> Swarm<FullBehaviour> {
    let keypair = libp2p::identity::Keypair::generate_ed25519();
    let peer_id = keypair.public().to_peer_id();
    let protocol = StreamProtocol::new("/test/1.0.0");

    SwarmBuilder::with_existing_identity(keypair.clone())
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .unwrap()
        .with_behaviour(|key| {
            let store = kad::store::MemoryStore::new(peer_id);
            let mut kademlia = kad::Behaviour::new(peer_id, store);
            kademlia.set_mode(Some(kad::Mode::Client));

            let message_id_fn = |message: &gossipsub::Message| {
                let hash = blake3::hash(&message.data);
                gossipsub::MessageId::from(hex::encode(&hash.as_bytes()[..16]))
            };
            let gossipsub_config = gossipsub::ConfigBuilder::default()
                .heartbeat_interval(Duration::from_secs(1)) // Fast heartbeat to generate events
                .validation_mode(gossipsub::ValidationMode::Strict)
                .message_id_fn(message_id_fn)
                .mesh_n(2)
                .mesh_n_low(1)
                .mesh_n_high(4)
                .mesh_outbound_min(1)
                .build()
                .unwrap();
            let gossipsub = gossipsub::Behaviour::new(
                gossipsub::MessageAuthenticity::Signed(keypair.clone()),
                gossipsub_config,
            )
            .unwrap();

            let identify = identify::Behaviour::new(identify::Config::new(
                "/test-identify/1.0.0".to_string(),
                key.public(),
            ));

            Ok(FullBehaviour {
                request_response: request_response::Behaviour::new(
                    [(protocol.clone(), ProtocolSupport::Full)],
                    request_response::Config::default()
                        .with_request_timeout(Duration::from_secs(30)),
                ),
                kademlia,
                gossipsub,
                identify,
            })
        })
        .unwrap()
        .with_swarm_config(|c| {
            c.with_idle_connection_timeout(Duration::from_secs(60))
                .with_notify_handler_buffer_size(std::num::NonZeroUsize::new(256).expect("256 > 0"))
        })
        .build()
}

/// Run N sequential request_response round-trips on FullBehaviour swarms.
async fn run_full_behaviour_exchanges(
    swarm1: &mut Swarm<FullBehaviour>,
    swarm2: &mut Swarm<FullBehaviour>,
    peer2_id: PeerId,
    count: usize,
    payload_size: usize,
) -> Result<Vec<Duration>, String> {
    let mut durations = Vec::new();
    let payload = vec![0xABu8; payload_size];

    for i in 0..count {
        let start = Instant::now();
        let request = RawRequest(payload.clone());

        let outbound_id = swarm1
            .behaviour_mut()
            .request_response
            .send_request(&peer2_id, request.clone());

        eprintln!(
            "  [round {}] sent request (outbound_id={:?}, payload={}B)",
            i + 1,
            outbound_id,
            payload_size
        );

        let result = timeout(Duration::from_secs(15), async {
            loop {
                tokio::select! {
                    event = swarm1.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    message: request_response::Message::Response { response, .. },
                                    ..
                                },
                            )) => {
                                assert_eq!(response.0.len(), payload_size);
                                break;
                            }
                            SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::OutboundFailure { error, .. },
                            )) => {
                                return Err(format!("OutboundFailure on round {}: {:?}", i + 1, error));
                            }
                            _ => {} // Ignore kademlia/gossipsub/identify events
                        }
                    }
                    event = swarm2.select_next_some() => {
                        if let SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    peer,
                                    message: request_response::Message::Request { request, channel, .. },
                                    ..
                                },
                            )) = event {
                                let resp = RawResponse(request.0);
                                if let Err(e) = swarm2.behaviour_mut().request_response.send_response(channel, resp) {
                                    return Err(format!("send_response failed on round {}: {:?}", i + 1, e));
                                }
                                eprintln!("  [round {}] swarm2 received+echoed from {:?}", i + 1, &peer.to_string()[..8]);
                        }
                    }
                }
            }
            Ok(())
        })
        .await;

        match result {
            Ok(Ok(())) => {
                let elapsed = start.elapsed();
                eprintln!("  [round {}] completed in {:?}", i + 1, elapsed);
                durations.push(elapsed);
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(format!("TIMEOUT on round {} (15s)", i + 1)),
        }
    }

    Ok(durations)
}

/// Connect two FullBehaviour swarms via TCP.
async fn connect_full_swarms(
    swarm1: &mut Swarm<FullBehaviour>,
    swarm2: &mut Swarm<FullBehaviour>,
) -> Result<(PeerId, PeerId), String> {
    let peer1_id = *swarm1.local_peer_id();
    let peer2_id = *swarm2.local_peer_id();

    let listen_addr: Multiaddr = "/ip4/127.0.0.1/tcp/0".parse().unwrap();
    swarm2.listen_on(listen_addr).unwrap();

    let swarm2_addr = timeout(Duration::from_secs(5), async {
        loop {
            if let SwarmEvent::NewListenAddr { address, .. } = swarm2.select_next_some().await {
                if address.to_string().contains("127.0.0.1") {
                    return address;
                }
            }
        }
    })
    .await
    .map_err(|_| "Timeout waiting for swarm2 listen address")?;

    eprintln!(
        "  swarm2 listening on {} (peer {})",
        swarm2_addr,
        &peer2_id.to_string()[..8]
    );

    swarm1
        .dial(swarm2_addr.clone())
        .map_err(|e| format!("Dial failed: {e}"))?;

    let connected = timeout(Duration::from_secs(10), async {
        let mut s1_connected = false;
        let mut s2_connected = false;
        loop {
            tokio::select! {
                event = swarm1.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer2_id {
                            s1_connected = true;
                            eprintln!("  swarm1 connected to swarm2");
                        }
                    }
                }
                event = swarm2.select_next_some() => {
                    if let SwarmEvent::ConnectionEstablished { peer_id, .. } = event {
                        if peer_id == peer1_id {
                            s2_connected = true;
                            eprintln!("  swarm2 connected to swarm1");
                        }
                    }
                }
            }
            if s1_connected && s2_connected {
                break;
            }
        }
    })
    .await;

    connected.map_err(|_| "Timeout waiting for connection")?;

    // Longer delay for identify/kademlia initial exchange
    tokio::time::sleep(Duration::from_millis(500)).await;

    Ok((peer1_id, peer2_id))
}

// ─── Test 1: TCP + Yamux (default 256KB window) — small payload ───

#[tokio::test]
async fn test_yamux_default_sequential_small() {
    eprintln!("\n=== TEST: TCP+Yamux (default 256KB window), 1KB payload, 5 rounds ===");
    let mut swarm1 = build_tcp_yamux_swarm();
    let mut swarm2 = build_tcp_yamux_swarm();
    let (_p1, p2) = connect_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5, "Expected 5 round-trips");
            for (i, d) in durations.iter().enumerate() {
                assert!(
                    d.as_secs() < 5,
                    "Round {} took {:?} — possible stall",
                    i + 1,
                    d
                );
            }
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 2: TCP + Yamux (16MB window) — small payload ───

#[tokio::test]
async fn test_yamux_16mb_sequential_small() {
    eprintln!("\n=== TEST: TCP+Yamux (16MB window), 1KB payload, 5 rounds ===");
    let mut swarm1 = build_tcp_yamux_16mb_swarm();
    let mut swarm2 = build_tcp_yamux_16mb_swarm();
    let (_p1, p2) = connect_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 3: TCP + Yamux (default) — large payload (tensor-sized) ───

#[tokio::test]
async fn test_yamux_default_sequential_large() {
    eprintln!("\n=== TEST: TCP+Yamux (default 256KB window), 4MB payload, 5 rounds ===");
    let mut swarm1 = build_tcp_yamux_swarm();
    let mut swarm2 = build_tcp_yamux_swarm();
    let (_p1, p2) = connect_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 4 * 1024 * 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 4: TCP + Yamux (16MB) — large payload ───

#[tokio::test]
async fn test_yamux_16mb_sequential_large() {
    eprintln!("\n=== TEST: TCP+Yamux (16MB window), 4MB payload, 5 rounds ===");
    let mut swarm1 = build_tcp_yamux_16mb_swarm();
    let mut swarm2 = build_tcp_yamux_16mb_swarm();
    let (_p1, p2) = connect_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 4 * 1024 * 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 5: QUIC only (no yamux) — small payload ───

#[tokio::test]
async fn test_quic_sequential_small() {
    eprintln!("\n=== TEST: QUIC (no yamux), 1KB payload, 5 rounds ===");
    let mut swarm1 = build_quic_swarm();
    let mut swarm2 = build_quic_swarm();
    let (_p1, p2) = connect_quic_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 6: QUIC only — large payload ───

#[tokio::test]
async fn test_quic_sequential_large() {
    eprintln!("\n=== TEST: QUIC (no yamux), 4MB payload, 5 rounds ===");
    let mut swarm1 = build_quic_swarm();
    let mut swarm2 = build_quic_swarm();
    let (_p1, p2) = connect_quic_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 5, 4 * 1024 * 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 7: TCP + Yamux — rapid fire (10 rounds, no delay) ───

#[tokio::test]
async fn test_yamux_rapid_fire() {
    eprintln!("\n=== TEST: TCP+Yamux (default), 1KB payload, 10 rapid rounds ===");
    let mut swarm1 = build_tcp_yamux_swarm();
    let mut swarm2 = build_tcp_yamux_swarm();
    let (_p1, p2) = connect_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 10, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 10);
            // Check no round took > 2s
            for (i, d) in durations.iter().enumerate() {
                assert!(
                    d.as_secs() < 2,
                    "Round {} took {:?} — stall detected",
                    i + 1,
                    d
                );
            }
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 8: QUIC — rapid fire ───

#[tokio::test]
async fn test_quic_rapid_fire() {
    eprintln!("\n=== TEST: QUIC (no yamux), 1KB payload, 10 rapid rounds ===");
    let mut swarm1 = build_quic_swarm();
    let mut swarm2 = build_quic_swarm();
    let (_p1, p2) = connect_quic_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_sequential_exchanges(&mut swarm1, &mut swarm2, p2, 10, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 10);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 9: Full behaviour with external command channel (simulates real daemon) ───
// The key difference: requests come from an external mpsc channel, not directly
// from the poll loop. This simulates how the real daemon sends tensor forwards.

#[tokio::test]
async fn test_full_behaviour_channel_driven() {
    eprintln!(
        "\n=== TEST: Full behaviour with channel-driven requests (simulates daemon), 4MB payload, 5 rounds ==="
    );
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();
    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let payload_size = 4 * 1024 * 1024;
    let (cmd_tx, mut cmd_rx) = tokio::sync::mpsc::channel::<(PeerId, Vec<u8>)>(16);

    // Spawn a "daemon" task that sends requests via channel with a 500ms compute delay
    let cmd_tx_clone = cmd_tx.clone();
    let p2_clone = p2;
    let payload_clone = vec![0xABu8; payload_size];
    tokio::spawn(async move {
        for i in 0..5 {
            // Simulate compute delay (model forward pass)
            tokio::time::sleep(Duration::from_millis(500)).await;
            eprintln!("  [daemon] sending request {} via channel", i + 1);
            if cmd_tx_clone
                .send((p2_clone, payload_clone.clone()))
                .await
                .is_err()
            {
                break;
            }
        }
    });

    let mut completed = 0u32;
    let mut durations = Vec::new();
    let mut round_start = Instant::now();

    let test_result = timeout(Duration::from_secs(60), async {
        loop {
            tokio::select! {
                // External command channel (like daemon's inbound_rx)
                Some((peer, data)) = cmd_rx.recv() => {
                    round_start = Instant::now();
                    let req = RawRequest(data);
                    let oid = swarm1.behaviour_mut().request_response.send_request(&peer, req);
                    eprintln!("  [swarm1] queued request via channel (outbound_id={:?})", oid);
                }
                // Swarm1 events
                event = swarm1.select_next_some() => {
                    match event {
                        SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                            request_response::Event::Message {
                                message: request_response::Message::Response { response, .. },
                                ..
                            },
                        )) => {
                            let elapsed = round_start.elapsed();
                            completed += 1;
                            assert_eq!(response.0.len(), payload_size);
                            eprintln!("  [swarm1] round {} response in {:?}", completed, elapsed);
                            durations.push(elapsed);
                            if completed >= 5 {
                                break;
                            }
                        }
                        SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                            request_response::Event::OutboundFailure { error, .. },
                        )) => {
                            return Err(format!("OutboundFailure: {:?}", error));
                        }
                        _ => {}
                    }
                }
                // Swarm2 events
                event = swarm2.select_next_some() => {
                    if let SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                            request_response::Event::Message {
                                message: request_response::Message::Request { request, channel, .. },
                                ..
                            },
                        )) = event {
                            // Simulate compute: sleep 100ms then echo
                            let resp = RawResponse(request.0);
                            if swarm2.behaviour_mut().request_response.send_response(channel, resp).is_err() {
                                eprintln!("  [swarm2] send_response failed");
                            }
                    }
                }
            }
        }
        Ok(())
    }).await;

    match test_result {
        Ok(Ok(())) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert_eq!(durations.len(), 5);
            for (i, d) in durations.iter().enumerate() {
                assert!(d.as_secs() < 5, "Round {} took {:?} — stall", i + 1, d);
            }
        }
        Ok(Err(e)) => panic!("FAILED: {e}"),
        Err(_) => panic!("TIMEOUT after 60s — completed {}/5 rounds", completed),
    }
}

// ─── Test 10a: Full behaviour with gossipsub subscriptions (heartbeat events) ───
// GossipSub heartbeat fires every 1s and generates NotifyBehaviour events for
// each subscribed topic. This is the closest simulation to production.

#[tokio::test]
async fn test_full_behaviour_with_gossipsub_topics() {
    eprintln!(
        "\n=== TEST: Full behaviour + gossipsub topics (heartbeat events), 4MB, 5 rounds ==="
    );
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();

    // Subscribe both swarms to multiple gossipsub topics (like production)
    let topics = vec![
        "swarm/models",
        "swarm/credits",
        "swarm/health",
        "swarm/identity",
        "swarm/pools",
    ];
    for topic_str in &topics {
        let topic = gossipsub::IdentTopic::new(*topic_str);
        swarm1.behaviour_mut().gossipsub.subscribe(&topic).unwrap();
        swarm2.behaviour_mut().gossipsub.subscribe(&topic).unwrap();
    }

    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    // Wait for gossipsub mesh to form (heartbeats need to exchange)
    eprintln!("  Waiting 3s for gossipsub mesh formation...");
    let mesh_wait = Instant::now();
    let mesh_deadline = Duration::from_secs(3);
    while mesh_wait.elapsed() < mesh_deadline {
        tokio::select! {
            _ = swarm1.select_next_some() => {}
            _ = swarm2.select_next_some() => {}
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    eprintln!("  Mesh formation complete, starting exchanges...");

    let result =
        run_full_behaviour_exchanges(&mut swarm1, &mut swarm2, p2, 5, 4 * 1024 * 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 10b: CPU-blocking compute on responder (simulates model forward) ───
// This is the KEY test — the responder blocks a Tokio worker thread for 700ms
// (simulating split_model.forward()) while both swarms share the same runtime.

#[tokio::test]
async fn test_full_behaviour_cpu_blocking_responder() {
    eprintln!(
        "\n=== TEST: Full behaviour + CPU-blocking responder (700ms), 4MB payload, 5 rounds ==="
    );
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();
    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let payload_size = 4 * 1024 * 1024;
    let payload = vec![0xABu8; payload_size];
    let mut durations = Vec::new();

    for i in 0..5 {
        let start = Instant::now();
        let req = RawRequest(payload.clone());
        let oid = swarm1
            .behaviour_mut()
            .request_response
            .send_request(&p2, req);
        eprintln!("  [round {}] sent (oid={:?})", i + 1, oid);

        let result = timeout(Duration::from_secs(30), async {
            loop {
                tokio::select! {
                    event = swarm1.select_next_some() => {
                        match event {
                            SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    message: request_response::Message::Response { response, .. },
                                    ..
                                },
                            )) => {
                                assert_eq!(response.0.len(), payload_size);
                                break;
                            }
                            SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::OutboundFailure { error, .. },
                            )) => {
                                return Err(format!("OutboundFailure: {:?}", error));
                            }
                            _ => {}
                        }
                    }
                    event = swarm2.select_next_some() => {
                        if let SwarmEvent::Behaviour(FullBehaviourEvent::RequestResponse(
                                request_response::Event::Message {
                                    message: request_response::Message::Request { request, channel, .. },
                                    ..
                                },
                            )) = event {
                                // SIMULATE: CPU-blocking model forward (700ms)
                                // This blocks the Tokio worker thread, exactly like
                                // split_model.forward() does in production.
                                eprintln!("  [round {}] swarm2: blocking thread for 700ms (model forward sim)", i + 1);
                                std::thread::sleep(Duration::from_millis(700));
                                let resp = RawResponse(request.0);
                                if swarm2.behaviour_mut().request_response.send_response(channel, resp).is_err() {
                                    return Err("send_response failed".to_string());
                                }
                                eprintln!("  [round {}] swarm2: response sent after blocking", i + 1);
                        }
                    }
                }
            }
            Ok(())
        }).await;

        match result {
            Ok(Ok(())) => {
                let elapsed = start.elapsed();
                eprintln!("  [round {}] completed in {:?}", i + 1, elapsed);
                durations.push(elapsed);
            }
            Ok(Err(e)) => panic!("FAILED round {}: {e}", i + 1),
            Err(_) => panic!("TIMEOUT round {} (30s)", i + 1),
        }
    }

    eprintln!("  ALL PASSED: {:?}", durations);
    for (i, d) in durations.iter().enumerate() {
        assert!(
            d.as_secs() < 10,
            "Round {} took {:?} — stall detected",
            i + 1,
            d
        );
    }
}

// ─── Test 10c: Full behaviour direct (kad+gossipsub+identify+rr) — small payload ───

#[tokio::test]
async fn test_full_behaviour_sequential_small() {
    eprintln!("\n=== TEST: Full behaviour (kad+gossipsub+identify+rr), 1KB payload, 5 rounds ===");
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();
    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_full_behaviour_exchanges(&mut swarm1, &mut swarm2, p2, 5, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
            for (i, d) in durations.iter().enumerate() {
                assert!(
                    d.as_secs() < 5,
                    "Round {} took {:?} — possible poll starvation",
                    i + 1,
                    d
                );
            }
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 10: Full behaviour — large payload (tensor-sized) ───

#[tokio::test]
async fn test_full_behaviour_sequential_large() {
    eprintln!("\n=== TEST: Full behaviour (kad+gossipsub+identify+rr), 4MB payload, 5 rounds ===");
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();
    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result =
        run_full_behaviour_exchanges(&mut swarm1, &mut swarm2, p2, 5, 4 * 1024 * 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 5);
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}

// ─── Test 11: Full behaviour — rapid fire 10 rounds ───

#[tokio::test]
async fn test_full_behaviour_rapid_fire() {
    eprintln!(
        "\n=== TEST: Full behaviour (kad+gossipsub+identify+rr), 1KB payload, 10 rapid rounds ==="
    );
    let mut swarm1 = build_full_behaviour_swarm();
    let mut swarm2 = build_full_behaviour_swarm();
    let (_p1, p2) = connect_full_swarms(&mut swarm1, &mut swarm2).await.unwrap();

    let result = run_full_behaviour_exchanges(&mut swarm1, &mut swarm2, p2, 10, 1024).await;
    match result {
        Ok(durations) => {
            eprintln!("  ALL PASSED: {:?}", durations);
            assert!(durations.len() == 10);
            for (i, d) in durations.iter().enumerate() {
                assert!(
                    d.as_secs() < 2,
                    "Round {} took {:?} — stall detected",
                    i + 1,
                    d
                );
            }
        }
        Err(e) => panic!("FAILED: {e}"),
    }
}
