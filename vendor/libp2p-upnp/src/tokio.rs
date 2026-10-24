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

use std::{
    collections::HashMap,
    error::Error,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    sync::{Arc, Mutex},
};

use futures::{
    channel::{mpsc, oneshot},
    SinkExt, StreamExt,
};
use igd_next::{
    aio::{tokio::Tokio, Gateway as IgdGateway},
    AddPortError, GetGenericPortMappingEntryError, PortMappingProtocol, RemovePortError,
    RequestError, SearchOptions,
};

pub use crate::behaviour::Behaviour;
use crate::{
    behaviour::{GatewayEvent, GatewayRequest, Mapping},
    choice::{self, Answer, Ours, Row, Table},
};

/// Every address this computer listens on, written by the behaviour as listen
/// addresses appear and read here at each mapping (`choice::Ours::own_ips`).
pub(crate) type OwnIps = Arc<Mutex<Vec<IpAddr>>>;

/// The description every mapping carries on the router (upstream's).
const MAPPING_DESCRIPTION: &str = "rust-libp2p mapping";

// TODO: remove when `IpAddr::is_global` stabilizes.
pub(crate) fn is_addr_global(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(ip) => {
            !(ip.octets()[0] == 0 // "This network"
                                    || ip.is_private()
                                    // code for Ipv4::is_shared()
                                    || (ip.octets()[0] == 100 && (ip.octets()[1] & 0b1100_0000 == 0b0100_0000))
                                    || ip.is_loopback()
                                    || ip.is_link_local()
                                    // addresses reserved for future protocols (`192.0.0.0/24`)
                                    ||(ip.octets()[0] == 192 && ip.octets()[1] == 0 && ip.octets()[2] == 0)
                                    || ip.is_documentation()
                                    // code for Ipv4::is_benchmarking()
                                    || (ip.octets()[0] == 198 && (ip.octets()[1] & 0xfe) == 18)
                                    // code for Ipv4::is_reserved()
                                    || (ip.octets()[0] & 240 == 240 && !ip.is_broadcast())
                                    || ip.is_broadcast())
        }
        IpAddr::V6(ip) => {
            !(ip.is_unspecified()
                                    || ip.is_loopback()
                                    // IPv4-mapped Address (`::ffff:0:0/96`)
                                    || matches!(ip.segments(), [0, 0, 0, 0, 0, 0xffff, _, _])
                                    // IPv4-IPv6 Translat. (`64:ff9b:1::/48`)
                                    || matches!(ip.segments(), [0x64, 0xff9b, 1, _, _, _, _, _])
                                    // Discard-Only Address Block (`100::/64`)
                                    || matches!(ip.segments(), [0x100, 0, 0, 0, _, _, _, _])
                                    // IETF Protocol Assignments (`2001::/23`)
                                    || (matches!(ip.segments(), [0x2001, b, _, _, _, _, _, _] if b < 0x200)
                                    && !(
                                    // Port Control Protocol Anycast (`2001:1::1`)
                                    u128::from_be_bytes(ip.octets()) == 0x2001_0001_0000_0000_0000_0000_0000_0001
                                    // Traversal Using Relays around NAT Anycast (`2001:1::2`)
                                    || u128::from_be_bytes(ip.octets()) == 0x2001_0001_0000_0000_0000_0000_0000_0002
                                    // AMT (`2001:3::/32`)
                                    || matches!(ip.segments(), [0x2001, 3, _, _, _, _, _, _])
                                    // AS112-v6 (`2001:4:112::/48`)
                                    || matches!(ip.segments(), [0x2001, 4, 0x112, _, _, _, _, _])
                                    // ORCHIDv2 (`2001:20::/28`)
                                    || matches!(ip.segments(), [0x2001, b, _, _, _, _, _, _] if (0x20..=0x2F).contains(&b))
                                ))
                                    // code for Ipv4::is_documentation()
                                    || (ip.segments()[0] == 0x2001) && (ip.segments()[1] == 0xdb8)
                                    // code for Ipv4::is_unique_local()
                                    || (ip.segments()[0] & 0xfe00) == 0xfc00
                                    // code for Ipv4::is_unicast_link_local()
                                    || (ip.segments()[0] & 0xffc0) == 0xfe80)
        }
    }
}

/// Interface that interacts with the inner gateway by messages,
/// `GatewayRequest`s and `GatewayEvent`s.
#[derive(Debug)]
pub(crate) struct Gateway {
    pub(crate) sender: mpsc::Sender<GatewayRequest>,
    pub(crate) receiver: mpsc::Receiver<GatewayEvent>,
    pub(crate) external_addr: IpAddr,
    /// The router's own address (SwarmLLM patch), to read `local_ip` again from.
    pub(crate) router: SocketAddr,
    /// This computer's address on the router's network (SwarmLLM patch): only a
    /// listen address on it can be forwarded to. `None` if it could not be read.
    pub(crate) local_ip: Option<IpAddr>,
}

/// The local address this computer reaches `router` from — what miniupnpc's
/// `UPNP_GetValidIGD` returns as `lanaddr`. A UDP `connect` sends nothing.
pub(crate) fn local_ip_towards(router: SocketAddr) -> Option<IpAddr> {
    let any: SocketAddr = match router {
        SocketAddr::V4(_) => (Ipv4Addr::UNSPECIFIED, 0).into(),
        SocketAddr::V6(_) => (Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    let socket = std::net::UdpSocket::bind(any).ok()?;
    socket.connect(router).ok()?;
    Some(socket.local_addr().ok()?.ip())
}

/// The router's port-mapping table.
async fn read_table(gateway: &IgdGateway<Tokio>) -> Table {
    read_table_until(gateway, |_| false).await
}

/// The router's table up to and including the first row `stop` accepts — one
/// request a row, so a removal at shutdown stops at its own row. Read past
/// empty slots: `choice::TableRead` says when it is over.
async fn read_table_until(gateway: &IgdGateway<Tokio>, stop: impl Fn(&Row) -> bool) -> Table {
    let mut read = choice::TableRead::default();
    while let Some(index) = read.next() {
        let answer = match gateway.get_generic_port_mapping_entry(index).await {
            Ok(entry) => Answer::Row(Row::from(entry)),
            Err(GetGenericPortMappingEntryError::SpecifiedArrayIndexInvalid) => Answer::NoEntry,
            // The request never got an answer (igd-next maps its 10 s timeout to
            // an I/O error): not the router refusing to list.
            Err(GetGenericPortMappingEntryError::RequestError(
                err @ (RequestError::IoError(_)
                | RequestError::HyperError(_)
                | RequestError::HyperClientError(_)),
            )) => {
                tracing::debug!(
                    index,
                    "the gateway did not answer a read of its port mappings: {err}"
                );
                Answer::Silent
            }
            Err(err) => {
                if index == 0 {
                    tracing::debug!("the gateway would not list its port mappings: {err}");
                }
                Answer::Refused
            }
        };
        read.answer(answer, &stop);
    }
    read.table()
}

const NO_ANSWER: &str = "the router did not answer a read of its port mappings";

/// Map `mapping` on the first external port no other computer holds, and return it.
async fn map_port(
    gateway: &IgdGateway<Tokio>,
    mapping: &Mapping,
    duration: u32,
    held: Option<u16>,
    own_ips: &[IpAddr],
) -> Result<u16, Box<dyn Error + Send + Sync>> {
    let internal = mapping.internal_addr;
    let ours = Ours {
        protocol: mapping.protocol,
        internal,
        own_ips,
    };
    let table = match read_table(gateway).await {
        Table::Listed(rows) => Some(rows),
        Table::Unlisted => None,
        Table::Unknown => return Err(NO_ANSWER.into()),
    };
    let mut last_err: Option<Box<dyn Error + Send + Sync>> = None;
    for port in choice::candidates(ours, held, table.as_deref()) {
        match gateway
            .add_port(
                mapping.protocol,
                port,
                internal,
                duration,
                MAPPING_DESCRIPTION,
            )
            .await
        {
            Ok(()) => {
                tracing::debug!(
                    protocol=%mapping.protocol,
                    internal_port=internal.port(),
                    external_port=port,
                    "UPnP: mapped"
                );
                return Ok(port);
            }
            // Another client holds it (718; some routers answer 501 — libtorrent's
            // rule), or the router maps only external = internal: try the next.
            Err(
                err @ (AddPortError::PortInUse
                | AddPortError::SamePortValuesRequired
                | AddPortError::RequestError(RequestError::ErrorCode(501, _))),
            ) => last_err = Some(err.into()),
            Err(err) => return Err(err.into()),
        }
    }
    Err(last_err.unwrap_or_else(|| "every external port asked for is another computer's".into()))
}

/// Remove the router's row for `mapping` at `port` — unless another computer
/// holds that port now (a router that replaces gave it away), which is theirs.
/// A router that did not answer the read is not asked to remove it either.
async fn unmap_port(
    gateway: &IgdGateway<Tokio>,
    ours: Ours<'_>,
    port: u16,
) -> Result<(), RemovePortError> {
    let protocol = ours.protocol;
    let this_row = |r: &Row| r.protocol == protocol && r.external_port == port;
    match read_table_until(gateway, this_row).await {
        Table::Listed(table) if choice::held_by_another(&table, ours, port) => {
            tracing::debug!(port, %protocol, "UPnP: not removing a mapping another computer holds now");
            return Ok(());
        }
        Table::Unknown => {
            return Err(RemovePortError::RequestError(RequestError::IoError(
                std::io::Error::new(std::io::ErrorKind::TimedOut, NO_ANSWER),
            )))
        }
        Table::Listed(_) | Table::Unlisted => {}
    }
    match gateway.remove_port(protocol, port).await {
        Ok(()) | Err(RemovePortError::NoSuchPortMapping) => Ok(()),
        Err(err) => Err(err),
    }
}

pub(crate) fn search_gateway(
    own_ips: OwnIps,
) -> oneshot::Receiver<Result<Gateway, Box<dyn Error + Send + Sync>>> {
    let (search_result_sender, search_result_receiver) = oneshot::channel();

    let (events_sender, mut task_receiver) = mpsc::channel(10);
    let (mut task_sender, events_queue) = mpsc::channel(0);

    tokio::spawn(async move {
        let gateway = match igd_next::aio::tokio::search_gateway(SearchOptions::default()).await {
            Ok(gateway) => gateway,
            Err(err) => {
                let _ = search_result_sender.send(Err(err.into()));
                return;
            }
        };

        let external_addr = match gateway.get_external_ip().await {
            Ok(addr) => addr,
            Err(err) => {
                let _ = search_result_sender.send(Err(err.into()));
                return;
            }
        };

        let local_ip = local_ip_towards(gateway.addr);

        // Check if receiver dropped.
        if search_result_sender
            .send(Ok(Gateway {
                sender: events_sender,
                receiver: events_queue,
                external_addr,
                router: gateway.addr,
                local_ip,
            }))
            .is_err()
        {
            return;
        }

        // The external port each mapping holds (go-nat's `ports`): renewed as it is,
        // and the one a removal deletes — never the internal port by assumption.
        let mut held: HashMap<(PortMappingProtocol, SocketAddr), u16> = HashMap::new();

        loop {
            // The task sender has dropped so we can return.
            let Some(req) = task_receiver.next().await else {
                return;
            };
            let own: Vec<IpAddr> = own_ips.lock().map(|v| v.clone()).unwrap_or_default();
            let event = match req {
                GatewayRequest::AddMapping { mapping, duration } => {
                    let key = (mapping.protocol, mapping.internal_addr);
                    match map_port(&gateway, &mapping, duration, held.get(&key).copied(), &own)
                        .await
                    {
                        Ok(external_port) => {
                            held.insert(key, external_port);
                            GatewayEvent::Mapped {
                                mapping,
                                external_port,
                            }
                        }
                        Err(err) => GatewayEvent::MapFailure(mapping, err),
                    }
                }
                GatewayRequest::RemoveMapping(mapping) => {
                    let key = (mapping.protocol, mapping.internal_addr);
                    match held.get(&key).copied() {
                        // Nothing of ours on the router.
                        None => GatewayEvent::Removed(mapping),
                        Some(port) => {
                            let ours = Ours {
                                protocol: mapping.protocol,
                                internal: mapping.internal_addr,
                                own_ips: &own,
                            };
                            match unmap_port(&gateway, ours, port).await {
                                Ok(()) => {
                                    held.remove(&key);
                                    GatewayEvent::Removed(mapping)
                                }
                                Err(err) => GatewayEvent::RemovalFailure(mapping, err.into()),
                            }
                        }
                    }
                }
            };
            // Gateway was dropped.
            if task_sender.send(event).await.is_err() {
                return;
            }
        }
    });

    search_result_receiver
}
