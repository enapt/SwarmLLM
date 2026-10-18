//! What each libp2p PROTOCOL puts on the wire — Kademlia, identify, gossip,
//! relay, AutoNAT and our own — counted per substream (`docs/FUTURE_WORK.md`
//! #91 (2e)).
//!
//! The transport total (`network::bandwidth::BandwidthMeter`) less gossip,
//! parts and inference left ~2.5 KB/s of an idle node's upload, and our own
//! messages (`DirectKindMeter`) named 0.19 KB/s of it (live node, 2026-10-18,
//! 1 → 2 h up). The rest belonged to libp2p's protocols, and `libp2p-metrics`
//! labels a byte only by its TRANSPORT (the multiaddr stack), never by the
//! stream's protocol. go-libp2p counts per protocol on the stream once it is
//! negotiated (`BandwidthCounter`, `GetBandwidthByProtocol`); rust-libp2p hands
//! a behaviour its stream with no hook, so this module reads the protocol off
//! the negotiation itself.
//!
//! Every substream opens with multistream-select: length-prefixed lines, each
//! ending in `\n` — the header `/multistream/1.0.0`, then the dialer's
//! proposal, which the listener echoes back or answers `na`
//! (multiformats/multistream-select). **The label is read from what THIS node
//! writes**: our proposal on a stream we open, our echo on one we accept. So it
//! is always a protocol this node speaks, and a peer cannot fill the table with
//! names of its own. The negotiation's bytes count to the protocol it chose; a
//! stream that ends before choosing one is [`UNNEGOTIATED`].
//!
//! Counted where `libp2p-metrics` counts — on each substream of each
//! connection's muxer, TCP, QUIC and relayed alike — so the rows add up to the
//! transport totals exactly, which is the known answer the tests check. Noise,
//! yamux and QUIC framing sit underneath, uncounted by both. A relayed
//! connection is counted twice by both as well: in its own protocols, and,
//! sealed, in the relay protocol of the connection it rides on.

use std::io::{self, IoSlice, IoSliceMut};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use futures::future::MapOk;
use futures::{AsyncRead, AsyncWrite, TryFutureExt};
use libp2p::core::muxing::{StreamMuxer, StreamMuxerEvent};
use libp2p::core::upgrade::{InboundConnectionUpgrade, OutboundConnectionUpgrade, UpgradeInfo};

/// Bytes on a substream that ended before this node named a protocol on it.
pub const UNNEGOTIATED: &str = "(unnegotiated)";
/// Bytes on a substream whose opening this node wrote was not a
/// multistream-select negotiation it could read. Never expected; a row here
/// means the reading below is wrong, not that a peer misbehaved.
pub const UNREADABLE: &str = "(unreadable)";
/// Bytes under a protocol that arrived after the table was full.
pub const OVERFLOW: &str = "(other)";

/// The most protocols the table keeps by name. A node speaks about a dozen;
/// the labels are our own writes, so this is a bound on a mistake, not a
/// defence against peers.
const MAX_PROTOCOLS: usize = 64;
/// How much of a stream's opening is kept to read the protocol from. The
/// header is 20 bytes and a protocol line at most 142 (multistream-select
/// caps a name at 140), with room for an `na` or two before it.
const SNIFF_CAP: usize = 512;
/// multistream-select's own cap on a protocol name.
const MAX_PROTOCOL_LEN: usize = 140;
const MULTISTREAM_HEADER: &[u8] = b"/multistream/1.0.0";

/// Bytes by protocol, since the daemon started.
#[derive(Debug, Default)]
pub struct ProtocolTrafficMeter {
    by_protocol: dashmap::DashMap<String, Arc<ProtocolCount>>,
}

#[derive(Debug, Default)]
pub struct ProtocolCount {
    streams: AtomicU64,
    sent_bytes: AtomicU64,
    recv_bytes: AtomicU64,
}

/// One protocol's share of this node's traffic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolTotals {
    pub protocol: String,
    /// Substreams that carried it.
    pub streams: u64,
    pub sent_bytes: u64,
    pub recv_bytes: u64,
}

impl ProtocolTrafficMeter {
    /// The counter a stream books to, once it has a name.
    fn count_for(&self, protocol: &str) -> Arc<ProtocolCount> {
        if let Some(count) = self.by_protocol.get(protocol) {
            return count.clone();
        }
        let key = if self.by_protocol.len() >= MAX_PROTOCOLS {
            OVERFLOW
        } else {
            protocol
        };
        self.by_protocol
            .entry(key.to_string())
            .or_default()
            .value()
            .clone()
    }

    /// Every protocol seen, the most SENT first — an idle node's upload is the
    /// question — then by what arrived, then by name, so equal rows keep their
    /// order between reads.
    pub fn totals(&self) -> Vec<ProtocolTotals> {
        let mut out: Vec<ProtocolTotals> = self
            .by_protocol
            .iter()
            .map(|e| ProtocolTotals {
                protocol: e.key().clone(),
                streams: e.value().streams.load(Ordering::Relaxed),
                sent_bytes: e.value().sent_bytes.load(Ordering::Relaxed),
                recv_bytes: e.value().recv_bytes.load(Ordering::Relaxed),
            })
            .collect();
        out.sort_by(|a, b| {
            b.sent_bytes
                .cmp(&a.sent_bytes)
                .then_with(|| b.recv_bytes.cmp(&a.recv_bytes))
                .then_with(|| a.protocol.cmp(&b.protocol))
        });
        out
    }
}

/// What the opening of a stream, as this node wrote it, says so far.
#[derive(Debug, PartialEq, Eq)]
enum Sniff<'a> {
    Named(&'a str),
    NeedMore,
    Unreadable,
}

/// The protocol this node named in what it wrote: the first line after the
/// header that is not `na` (a refused proposal we answered) or `ls`.
fn protocol_named(written: &[u8]) -> Sniff<'_> {
    let mut rest = written;
    loop {
        // An unsigned varint of at most two bytes: multistream-select's own
        // limit, and a protocol line never needs more.
        let (len, used) = match rest {
            [] => return Sniff::NeedMore,
            [b, ..] if b & 0x80 == 0 => (*b as usize, 1),
            [_] => return Sniff::NeedMore,
            [lo, hi, ..] if hi & 0x80 == 0 => ((*lo & 0x7f) as usize | (*hi as usize) << 7, 2),
            _ => return Sniff::Unreadable,
        };
        if len == 0 || len > MAX_PROTOCOL_LEN + 1 {
            return Sniff::Unreadable;
        }
        let Some(line) = rest.get(used..used + len) else {
            return Sniff::NeedMore;
        };
        rest = &rest[used + len..];
        let Some(text) = line.strip_suffix(b"\n") else {
            return Sniff::Unreadable;
        };
        if text == MULTISTREAM_HEADER || text == b"na" || text == b"ls" {
            continue;
        }
        return match std::str::from_utf8(text) {
            Ok(name) if name.starts_with('/') && name.bytes().all(|b| b.is_ascii_graphic()) => {
                Sniff::Named(name)
            }
            _ => Sniff::Unreadable,
        };
    }
}

/// A multiplexer upgrade whose muxer counts its substreams by protocol.
#[derive(Clone)]
pub struct MeteredMuxUpgrade<U> {
    inner: U,
    meter: Arc<ProtocolTrafficMeter>,
}

/// The form the swarm builder takes a multiplexer in (`with_tcp`,
/// `with_relay_client`): `metered_muxer(libp2p::yamux::Config::default, meter)`.
pub fn metered_muxer<U>(
    make: impl FnOnce() -> U,
    meter: Arc<ProtocolTrafficMeter>,
) -> impl FnOnce() -> MeteredMuxUpgrade<U> {
    move || MeteredMuxUpgrade {
        inner: make(),
        meter,
    }
}

impl<U: UpgradeInfo> UpgradeInfo for MeteredMuxUpgrade<U> {
    type Info = U::Info;
    type InfoIter = U::InfoIter;

    fn protocol_info(&self) -> Self::InfoIter {
        self.inner.protocol_info()
    }
}

type Wrap<M> = Box<dyn FnOnce(M) -> MeteredMuxer<M> + Send>;

impl<C, U> InboundConnectionUpgrade<C> for MeteredMuxUpgrade<U>
where
    U: InboundConnectionUpgrade<C>,
{
    type Output = MeteredMuxer<U::Output>;
    type Error = U::Error;
    type Future = MapOk<U::Future, Wrap<U::Output>>;

    fn upgrade_inbound(self, socket: C, info: Self::Info) -> Self::Future {
        let meter = self.meter;
        self.inner
            .upgrade_inbound(socket, info)
            .map_ok(Box::new(move |muxer| MeteredMuxer::new(muxer, meter)))
    }
}

impl<C, U> OutboundConnectionUpgrade<C> for MeteredMuxUpgrade<U>
where
    U: OutboundConnectionUpgrade<C>,
{
    type Output = MeteredMuxer<U::Output>;
    type Error = U::Error;
    type Future = MapOk<U::Future, Wrap<U::Output>>;

    fn upgrade_outbound(self, socket: C, info: Self::Info) -> Self::Future {
        let meter = self.meter;
        self.inner
            .upgrade_outbound(socket, info)
            .map_ok(Box::new(move |muxer| MeteredMuxer::new(muxer, meter)))
    }
}

/// A connection's muxer, every substream of it counted by protocol. QUIC has
/// no multiplexer upgrade to wrap, so its connection is wrapped directly
/// (`network::manager`'s swarm build).
pub struct MeteredMuxer<M> {
    inner: M,
    meter: Arc<ProtocolTrafficMeter>,
}

impl<M> MeteredMuxer<M> {
    pub fn new(inner: M, meter: Arc<ProtocolTrafficMeter>) -> Self {
        Self { inner, meter }
    }
}

impl<M> StreamMuxer for MeteredMuxer<M>
where
    M: StreamMuxer + Unpin,
    M::Substream: Unpin,
{
    type Substream = MeteredStream<M::Substream>;
    type Error = M::Error;

    fn poll_inbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();
        let inner = ready!(Pin::new(&mut this.inner).poll_inbound(cx))?;
        Poll::Ready(Ok(MeteredStream::new(inner, this.meter.clone())))
    }

    fn poll_outbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();
        let inner = ready!(Pin::new(&mut this.inner).poll_outbound(cx))?;
        Poll::Ready(Ok(MeteredStream::new(inner, this.meter.clone())))
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<StreamMuxerEvent, Self::Error>> {
        Pin::new(&mut self.get_mut().inner).poll(cx)
    }
}

/// Whether a stream knows its protocol yet.
enum Label {
    /// Not yet: what this node wrote so far (up to [`SNIFF_CAP`]), and the
    /// bytes each way waiting to be booked once it does.
    Unknown {
        written: Vec<u8>,
        sent: u64,
        recv: u64,
    },
    Known(Arc<ProtocolCount>),
}

/// One substream, its bytes booked to the protocol it negotiated.
pub struct MeteredStream<S> {
    inner: S,
    meter: Arc<ProtocolTrafficMeter>,
    label: Label,
}

impl<S> MeteredStream<S> {
    fn new(inner: S, meter: Arc<ProtocolTrafficMeter>) -> Self {
        Self {
            inner,
            meter,
            label: Label::Unknown {
                written: Vec::new(),
                sent: 0,
                recv: 0,
            },
        }
    }

    /// Book `n` bytes this node wrote, `chunks` being what they were.
    fn note_sent<'a>(&mut self, n: usize, chunks: impl Iterator<Item = &'a [u8]>) {
        let named = match &mut self.label {
            Label::Known(count) => {
                count.sent_bytes.fetch_add(n as u64, Ordering::Relaxed);
                return;
            }
            Label::Unknown { written, sent, .. } => {
                *sent += n as u64;
                let mut left = n;
                for chunk in chunks {
                    let take = chunk.len().min(left).min(SNIFF_CAP - written.len());
                    written.extend_from_slice(&chunk[..take]);
                    left -= chunk.len().min(left);
                    if left == 0 || written.len() == SNIFF_CAP {
                        break;
                    }
                }
                match protocol_named(written) {
                    Sniff::Named(name) => self.meter.count_for(name),
                    Sniff::NeedMore if written.len() < SNIFF_CAP => return,
                    Sniff::NeedMore | Sniff::Unreadable => self.meter.count_for(UNREADABLE),
                }
            }
        };
        self.book(named);
    }

    fn note_recv(&mut self, n: usize) {
        match &mut self.label {
            Label::Known(count) => {
                count.recv_bytes.fetch_add(n as u64, Ordering::Relaxed);
            }
            Label::Unknown { recv, .. } => *recv += n as u64,
        }
    }

    /// Name the stream, booking what it carried before it had a name.
    fn book(&mut self, count: Arc<ProtocolCount>) {
        if let Label::Unknown { sent, recv, .. } = self.label {
            count.streams.fetch_add(1, Ordering::Relaxed);
            count.sent_bytes.fetch_add(sent, Ordering::Relaxed);
            count.recv_bytes.fetch_add(recv, Ordering::Relaxed);
        }
        self.label = Label::Known(count);
    }
}

impl<S> Drop for MeteredStream<S> {
    fn drop(&mut self) {
        if let Label::Unknown { sent, recv, .. } = self.label {
            if sent + recv > 0 {
                let count = self.meter.count_for(UNNEGOTIATED);
                self.book(count);
            }
        }
    }
}

impl<S: AsyncRead + Unpin> AsyncRead for MeteredStream<S> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let n = ready!(Pin::new(&mut this.inner).poll_read(cx, buf))?;
        this.note_recv(n);
        Poll::Ready(Ok(n))
    }

    fn poll_read_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &mut [IoSliceMut<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let n = ready!(Pin::new(&mut this.inner).poll_read_vectored(cx, bufs))?;
        this.note_recv(n);
        Poll::Ready(Ok(n))
    }
}

impl<S: AsyncWrite + Unpin> AsyncWrite for MeteredStream<S> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let n = ready!(Pin::new(&mut this.inner).poll_write(cx, buf))?;
        this.note_sent(n, std::iter::once(buf));
        Poll::Ready(Ok(n))
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let n = ready!(Pin::new(&mut this.inner).poll_write_vectored(cx, bufs))?;
        this.note_sent(n, bufs.iter().map(|b| &**b));
        Poll::Ready(Ok(n))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_close(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A multistream-select line as it goes on the wire.
    fn line(text: &str) -> Vec<u8> {
        let mut out = vec![(text.len() + 1) as u8];
        out.extend_from_slice(text.as_bytes());
        out.push(b'\n');
        out
    }

    #[test]
    fn the_protocol_is_the_first_line_after_the_header_that_is_not_a_refusal() {
        let header = line("/multistream/1.0.0");
        assert_eq!(protocol_named(&header), Sniff::NeedMore);
        assert_eq!(protocol_named(&header[..5]), Sniff::NeedMore);

        let mut proposal = header.clone();
        proposal.extend(line("/swarmllm/kad/1.0.0"));
        proposal.extend_from_slice(b"\x00\x01data that follows a lazy proposal");
        assert_eq!(
            protocol_named(&proposal),
            Sniff::Named("/swarmllm/kad/1.0.0")
        );

        // A listener that refused the first proposal and took the second.
        let mut echo = header.clone();
        echo.extend(line("na"));
        echo.extend(line("/meshsub/1.1.0"));
        assert_eq!(protocol_named(&echo), Sniff::Named("/meshsub/1.1.0"));

        // Unreadable at once: a line with no newline, a length past two
        // bytes, a name that is not a path. (Anything else that is not a
        // negotiation reads as "need more" until `SNIFF_CAP` and is booked
        // as `UNREADABLE` then.)
        assert_eq!(protocol_named(b"\x03abc"), Sniff::Unreadable);
        assert_eq!(protocol_named(b"\x80\x80\x01"), Sniff::Unreadable);
        assert_eq!(protocol_named(&line("no-slash")), Sniff::Unreadable);
    }

    #[test]
    fn a_stream_that_ends_unnamed_is_still_counted() {
        let meter = Arc::new(ProtocolTrafficMeter::default());
        let mut stream = MeteredStream::new((), meter.clone());
        let header = line("/multistream/1.0.0");
        stream.note_sent(header.len(), std::iter::once(&header[..]));
        stream.note_recv(7);
        drop(stream);
        assert_eq!(
            meter.totals(),
            vec![ProtocolTotals {
                protocol: UNNEGOTIATED.into(),
                streams: 1,
                sent_bytes: header.len() as u64,
                recv_bytes: 7,
            }]
        );
    }

    /// Bytes that cross before the name is known — the dialer's lazy data, the
    /// listener's header — are booked to it, not lost.
    #[test]
    fn what_crossed_before_the_name_was_known_is_booked_to_it() {
        let meter = Arc::new(ProtocolTrafficMeter::default());
        let mut stream = MeteredStream::new((), meter.clone());
        stream.note_recv(30);
        let header = line("/multistream/1.0.0");
        let echo = line("/ipfs/id/1.0.0");
        stream.note_sent(header.len(), std::iter::once(&header[..]));
        // Vectored, split mid-line: the reading spans the writes.
        let (a, b) = echo.split_at(4);
        stream.note_sent(echo.len(), [a, b].into_iter());
        stream.note_sent(100, std::iter::once(&[0u8; 100][..]));
        stream.note_recv(50);
        drop(stream);
        assert_eq!(
            meter.totals(),
            vec![ProtocolTotals {
                protocol: "/ipfs/id/1.0.0".into(),
                streams: 1,
                sent_bytes: (header.len() + echo.len() + 100) as u64,
                recv_bytes: 80,
            }]
        );
    }

    /// The node's own swarm build meters every transport. The test below
    /// builds its swarms by hand, so nothing else would notice the node's
    /// build losing one — and an unmetered transport reads as rows that no
    /// longer add up, not as a missing row.
    #[test]
    fn every_transport_the_node_builds_is_metered_by_protocol() {
        let build = include_str!("manager/mod.rs");
        assert_eq!(
            build.matches("protocol_traffic::metered_muxer(").count(),
            2,
            "TCP and relayed circuits take the metered yamux"
        );
        assert_eq!(
            build
                .matches("protocol_traffic::MeteredMuxer::new(")
                .count(),
            1,
            "QUIC's connection is wrapped where it is added"
        );
        assert!(
            !build.contains(".with_quic()"),
            "`with_quic()` adds QUIC unmetered"
        );
    }

    /// Two real swarms, over TCP and over QUIC, wired the way the node wires
    /// its own: every protocol they negotiate is named, and the rows add up to
    /// what `libp2p-metrics` counted on the same connection — the known
    /// answer.
    #[tokio::test]
    async fn the_rows_add_up_to_the_transport_totals_on_tcp_and_quic() {
        for quic in [false, true] {
            rows_add_up(quic).await;
        }
    }

    #[derive(libp2p::swarm::NetworkBehaviour)]
    struct Probe {
        identify: libp2p::identify::Behaviour,
        kad: libp2p::kad::Behaviour<libp2p::kad::store::MemoryStore>,
    }

    /// A swarm built the way `network::manager` builds the node's: yamux
    /// wrapped through the builder's multiplexer argument, QUIC wrapped where
    /// it is added, `libp2p-metrics` over both.
    fn probe_swarm(
        meter: Arc<ProtocolTrafficMeter>,
        transport_meter: &crate::network::bandwidth::BandwidthMeter,
    ) -> libp2p::Swarm<Probe> {
        use libp2p::core::Transport as _;
        let behaviour = |key: &libp2p::identity::Keypair| {
            let peer = key.public().to_peer_id();
            let mut kad = libp2p::kad::Behaviour::with_config(
                peer,
                libp2p::kad::store::MemoryStore::new(peer),
                libp2p::kad::Config::new(libp2p::StreamProtocol::new("/probe/kad/1.0.0")),
            );
            kad.set_mode(Some(libp2p::kad::Mode::Server));
            Probe {
                identify: libp2p::identify::Behaviour::new(libp2p::identify::Config::new(
                    "/probe/1.0.0".into(),
                    key.public(),
                )),
                kad,
            }
        };
        let quic_meter = meter.clone();
        transport_meter.arm(|registry| {
            libp2p::SwarmBuilder::with_new_identity()
                .with_tokio()
                .with_tcp(
                    libp2p::tcp::Config::default(),
                    libp2p::noise::Config::new,
                    metered_muxer(libp2p::yamux::Config::default, meter),
                )
                .expect("tcp")
                .with_other_transport(|key| {
                    libp2p::quic::tokio::Transport::new(libp2p::quic::Config::new(key)).map(
                        move |(peer, connection), _| {
                            (peer, MeteredMuxer::new(connection, quic_meter.clone()))
                        },
                    )
                })
                .unwrap_or_else(|never| match never {})
                .with_bandwidth_metrics(registry)
                .with_behaviour(behaviour)
                .expect("behaviour")
                .with_swarm_config(|c| {
                    c.with_idle_connection_timeout(std::time::Duration::from_secs(60))
                })
                .build()
        })
    }

    async fn rows_add_up(quic: bool) {
        use futures::StreamExt;
        use libp2p::swarm::SwarmEvent;

        let (a_meter, b_meter) = (
            Arc::new(ProtocolTrafficMeter::default()),
            Arc::new(ProtocolTrafficMeter::default()),
        );
        let (a_bw, b_bw) = (
            crate::network::bandwidth::BandwidthMeter::new(),
            crate::network::bandwidth::BandwidthMeter::new(),
        );
        let mut a = probe_swarm(a_meter.clone(), &a_bw);
        let mut b = probe_swarm(b_meter.clone(), &b_bw);
        let listen: libp2p::Multiaddr = if quic {
            "/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap()
        } else {
            "/ip4/127.0.0.1/tcp/0".parse().unwrap()
        };
        b.listen_on(listen).unwrap();
        let addr = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = b.select_next_some().await {
                break address;
            }
        };
        let b_id = *b.local_peer_id();
        a.behaviour_mut().kad.add_address(&b_id, addr.clone());
        a.dial(addr).unwrap();

        // Run until A has heard B's identify AND A's Kademlia query has an
        // answer, so both protocols have crossed in both directions.
        let (mut asked, mut identified, mut queried) = (false, false, false);
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
        while !(identified && queried) {
            tokio::select! {
                ev = a.select_next_some() => match ev {
                    SwarmEvent::ConnectionEstablished { .. } if !asked => {
                        asked = true;
                        a.behaviour_mut().kad.get_closest_peers(b_id);
                    }
                    SwarmEvent::Behaviour(ProbeEvent::Identify(
                        libp2p::identify::Event::Received { .. },
                    )) => identified = true,
                    SwarmEvent::Behaviour(ProbeEvent::Kad(
                        libp2p::kad::Event::OutboundQueryProgressed { step, .. },
                    )) if step.last => queried = true,
                    _ => {}
                },
                _ = b.select_next_some() => {}
                _ = tokio::time::sleep_until(deadline) => panic!(
                    "quic={quic}: identified={identified} queried={queried} within 20 s"
                ),
            }
        }
        // Let both sides finish what they were writing.
        let settle = tokio::time::Instant::now() + std::time::Duration::from_millis(500);
        loop {
            tokio::select! {
                _ = a.select_next_some() => {}
                _ = b.select_next_some() => {}
                _ = tokio::time::sleep_until(settle) => break,
            }
        }

        for (who, meter, bw) in [("A", &a_meter, &a_bw), ("B", &b_meter, &b_bw)] {
            // Each connection runs in its own task, so a byte can still be on
            // its way between the two readings: read until they agree.
            let mut tries = 0;
            let (rows, sent, recv, transport) = loop {
                let rows = meter.totals();
                let transport = bw.totals().expect("the transport counted the connection");
                let sent: u64 = rows.iter().map(|r| r.sent_bytes).sum();
                let recv: u64 = rows.iter().map(|r| r.recv_bytes).sum();
                tries += 1;
                if (sent, recv) == (transport.outbound_bytes, transport.inbound_bytes)
                    || tries == 20
                {
                    break (rows, sent, recv, transport);
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            };
            let names: Vec<&str> = rows.iter().map(|r| r.protocol.as_str()).collect();
            for wanted in ["/ipfs/id/1.0.0", "/probe/kad/1.0.0"] {
                let row = rows
                    .iter()
                    .find(|r| r.protocol == wanted)
                    .unwrap_or_else(|| panic!("quic={quic} {who}: no {wanted} row in {names:?}"));
                assert!(
                    row.streams >= 1 && row.sent_bytes > 0 && row.recv_bytes > 0,
                    "quic={quic} {who}: {row:?}"
                );
            }
            assert!(
                !names.contains(&UNREADABLE),
                "quic={quic} {who}: an opening this node wrote did not read: {rows:?}"
            );
            assert_eq!(
                (sent, recv),
                (transport.outbound_bytes, transport.inbound_bytes),
                "quic={quic} {who}: the protocols must add up to the transport ({rows:?})"
            );
        }
    }
}
