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

mod connecting;
mod stream;

use std::{
    pin::Pin,
    task::{Context, Poll},
};

pub use connecting::Connecting;
use futures::{future::BoxFuture, FutureExt};
use libp2p_core::muxing::{StreamMuxer, StreamMuxerEvent};
pub use stream::Stream;

use crate::{ConnectionError, Error};

/// State for a single opened QUIC connection.
pub struct Connection {
    /// Underlying connection.
    connection: quinn::Connection,
    /// Future for accepting a new incoming bidirectional stream.
    incoming: Option<
        BoxFuture<'static, Result<(quinn::SendStream, quinn::RecvStream), quinn::ConnectionError>>,
    >,
    /// Future for opening a new outgoing bidirectional stream.
    outgoing: Option<
        BoxFuture<'static, Result<(quinn::SendStream, quinn::RecvStream), quinn::ConnectionError>>,
    >,
    /// Future to wait for the connection to be closed.
    closing: Option<BoxFuture<'static, quinn::ConnectionError>>,
}

impl Connection {
    /// Build a [`Connection`] from raw components.
    ///
    /// This function assumes that the [`quinn::Connection`] is completely fresh and none of
    /// its methods has ever been called. Failure to comply might lead to logic errors and panics.
    fn new(connection: quinn::Connection) -> Self {
        Self {
            connection,
            incoming: None,
            outgoing: None,
            closing: None,
        }
    }
}

impl StreamMuxer for Connection {
    type Substream = Stream;
    type Error = Error;

    fn poll_inbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();

        let incoming = this.incoming.get_or_insert_with(|| {
            let connection = this.connection.clone();
            async move { connection.accept_bi().await }.boxed()
        });

        let (send, recv) = futures::ready!(incoming.poll_unpin(cx)).map_err(ConnectionError)?;
        this.incoming.take();
        let stream = Stream::new(send, recv);
        Poll::Ready(Ok(stream))
    }

    fn poll_outbound(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<Self::Substream, Self::Error>> {
        let this = self.get_mut();

        let outgoing = this.outgoing.get_or_insert_with(|| {
            let connection = this.connection.clone();
            async move { connection.open_bi().await }.boxed()
        });

        let (send, recv) = futures::ready!(outgoing.poll_unpin(cx)).map_err(ConnectionError)?;
        this.outgoing.take();
        let stream = Stream::new(send, recv);
        Poll::Ready(Ok(stream))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Result<StreamMuxerEvent, Self::Error>> {
        // TODO: If connection migration is enabled (currently disabled) address
        // change on the connection needs to be handled.
        Poll::Pending
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();

        let closing = this.closing.get_or_insert_with(|| {
            this.connection.close(From::from(0u32), &[]);
            let connection = this.connection.clone();
            async move { connection.closed().await }.boxed()
        });

        match futures::ready!(closing.poll_unpin(cx)) {
            // Expected error given that `connection.close` was called above.
            quinn::ConnectionError::LocallyClosed => {}
            error => return Poll::Ready(Err(Error::Connection(ConnectionError(error)))),
        };

        Poll::Ready(Ok(()))
    }
}

/// SwarmLLM patch: a connection runs the congestion controller its `Config`
/// names (`config::Congestion`) — asked of the live quinn connection.
#[cfg(all(test, feature = "tokio"))]
mod congestion_tests {
    use futures::{future, StreamExt};
    use libp2p_core::{
        transport::{DialOpts, ListenerId, PortUse, TransportEvent},
        Endpoint, Transport,
    };

    use crate::{Config, Congestion, GenTransport};

    async fn dialled_with(congestion: Congestion) -> super::Connection {
        let transport = || {
            let keypair = libp2p_identity::Keypair::generate_ed25519();
            GenTransport::<crate::tokio::Provider>::new(
                Config::new(&keypair).congestion(congestion),
            )
            .boxed()
        };
        let mut listener = transport();
        let mut dialer = transport();
        listener
            .listen_on(
                ListenerId::next(),
                "/ip4/127.0.0.1/udp/0/quic-v1".parse().unwrap(),
            )
            .unwrap();
        let addr = match listener.next().await {
            Some(TransportEvent::NewAddress { listen_addr, .. }) => listen_addr,
            e => panic!("{e:?}"),
        };
        let accepted = async {
            let (upgrade, _) = listener.select_next_some().await.into_incoming().unwrap();
            upgrade.await.unwrap()
        };
        let dial = dialer
            .dial(
                addr,
                DialOpts {
                    role: Endpoint::Dialer,
                    port_use: PortUse::Reuse,
                },
            )
            .unwrap();
        let dialled = async {
            match future::select(dial, dialer.next()).await {
                future::Either::Left((Ok((_, connection)), _)) => connection,
                future::Either::Left((Err(e), _)) => panic!("dial failed: {e}"),
                future::Either::Right(_) => panic!("the dialer's transport ended first"),
            }
        };
        let ((_, _accepted), connection) = future::join(accepted, dialled).await;
        connection
    }

    fn runs<T: 'static>(connection: &super::Connection) -> bool {
        connection
            .connection
            .congestion_state()
            .into_any()
            .is::<T>()
    }

    #[tokio::test]
    async fn a_connection_runs_the_congestion_controller_its_config_names() {
        let bbr = dialled_with(Congestion::Bbr).await;
        assert!(runs::<quinn::congestion::Bbr>(&bbr), "Congestion::Bbr did not run BBR");
        let cubic = dialled_with(Congestion::Cubic).await;
        assert!(runs::<quinn::congestion::Cubic>(&cubic), "the default is not quinn's Cubic");
    }
}
