// SwarmLLM patch (2026-10-24): which EXTERNAL port a mapping asks the router for.
//
// Upstream asks for external = internal, always. Two nodes behind one router then
// want the same port. A router that refuses (UPnP error 718) leaves the second node
// unmapped. A router that silently REPLACES the first client's mapping (a SKYWORTH
// GPON box did: the second node's request took 8810/8800 from the first) hands the
// port to whichever node renewed last, while BOTH advertise it as theirs — so peers
// dialling one reach the other, every half hour in turn.
//
// So the router's table is read first and a port another computer holds is never
// asked for. Order: the port this mapping already holds; a port the router already
// forwards to this computer (a run that was killed — that router keeps a lease past
// its hour); the internal port; then ports derived from the address, so a restart
// asks for the same one again.
//
// "This computer" is any of its own listen addresses at the same internal port: a
// node before this patch also mapped WSL's 10.255.255.254 (or a Docker bridge), and
// the router kept that row. No other computer can hold our address, and no other
// node here can listen on our port, so the row is ours to take back.
//
// How others do it: go-libp2p (go-nat `upnp.go`) maps a RANDOM external port and
// remembers it for the run; libtorrent (`upnp.cpp`) asks for the caller's port and
// moves to `40000 + random(10000)` on 718 or 501. Neither reads the table, so
// neither sees a router that replaces. A refusal is still honoured here the
// libtorrent way: the next candidate is asked (`tokio.rs`).

use std::net::{IpAddr, SocketAddr};

use igd_next::PortMappingProtocol;

/// How many derived ports are tried after the internal one.
pub(crate) const DERIVED_CANDIDATES: u8 = 8;

/// Derived ports fall in libtorrent's range for a moved mapping.
const DERIVED_BASE: u16 = 40_000;
const DERIVED_SPAN: u32 = 10_000;

/// The most rows read from a router's table; a longer table is read that far.
pub(crate) const MAX_TABLE_ROWS: u32 = 128;

/// How many "no such entry" answers (713) in a row end a read of the table. A
/// router may keep its rows in fixed SLOTS and answer 713 for an empty one — the
/// same code as the end of the table. The 2026-10-24 router does: after the row
/// in slot 0 was deleted its listing read as empty while six rows remained, and
/// a new row fills the first empty slot (measured with throwaway rows). Stopping
/// at the first 713 would read another computer's port as free. ~3 ms a request
/// on that router.
pub(crate) const GAPS_THAT_END_THE_TABLE: u32 = 16;

/// The router's answer for one index of its table.
#[derive(Debug)]
pub(crate) enum Answer {
    Row(Row),
    /// 713: an empty slot, or past the end.
    NoEntry,
    /// Anything else (at the first index: the router will not list its table).
    Error,
}

/// A read of the router's table, one index at a time (`tokio.rs` asks).
#[derive(Debug, Default)]
pub(crate) struct TableRead {
    rows: Vec<Row>,
    index: u32,
    gaps: u32,
    done: bool,
    unlisted: bool,
}

impl TableRead {
    /// The index to ask for next; `None` once the read is over.
    pub(crate) fn next(&self) -> Option<u32> {
        (!self.done && self.index < MAX_TABLE_ROWS && self.gaps < GAPS_THAT_END_THE_TABLE)
            .then_some(self.index)
    }

    /// The answer for the index `next` gave; a row `found` accepts ends the read.
    pub(crate) fn answer(&mut self, answer: Answer, found: impl Fn(&Row) -> bool) {
        match answer {
            Answer::Row(row) => {
                self.gaps = 0;
                self.done = found(&row);
                self.rows.push(row);
            }
            Answer::NoEntry => self.gaps += 1,
            Answer::Error if self.index == 0 => {
                self.unlisted = true;
                self.done = true;
            }
            Answer::Error => self.gaps += 1,
        }
        self.index += 1;
    }

    /// The rows read, or `None` if the router would not list them.
    pub(crate) fn rows(self) -> Option<Vec<Row>> {
        (!self.unlisted).then_some(self.rows)
    }
}

/// One row of the router's port-mapping table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Row {
    pub(crate) external_port: u16,
    pub(crate) protocol: PortMappingProtocol,
    /// An address, or a host name on some routers (never equal to ours, then).
    pub(crate) internal_client: String,
    pub(crate) internal_port: u16,
}

impl From<igd_next::PortMappingEntry> for Row {
    fn from(e: igd_next::PortMappingEntry) -> Self {
        Row {
            external_port: e.external_port,
            protocol: e.protocol,
            internal_client: e.internal_client,
            internal_port: e.internal_port,
        }
    }
}

/// The mapping being made, and this computer's other addresses.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ours<'a> {
    pub(crate) protocol: PortMappingProtocol,
    pub(crate) internal: SocketAddr,
    /// Every address this computer listens on (the router-facing one too).
    pub(crate) own_ips: &'a [IpAddr],
}

impl Row {
    fn forwards_to(&self, ours: Ours<'_>) -> bool {
        self.protocol == ours.protocol
            && self.internal_port == ours.internal.port()
            && self
                .internal_client
                .trim()
                .parse::<IpAddr>()
                .is_ok_and(|ip| ip == ours.internal.ip() || ours.own_ips.contains(&ip))
    }
}

/// Does the router forward `external_port` to another computer?
pub(crate) fn held_by_another(table: &[Row], ours: Ours<'_>, external_port: u16) -> bool {
    table.iter().any(|r| {
        r.protocol == ours.protocol && r.external_port == external_port && !r.forwards_to(ours)
    })
}

/// External ports to ask for, best first. `held` is the port this mapping holds
/// now (a renewal); `table` is `None` when the router would not list its mappings,
/// and then only a refusal can say a port is taken.
pub(crate) fn candidates(ours: Ours<'_>, held: Option<u16>, table: Option<&[Row]>) -> Vec<u16> {
    fn push(out: &mut Vec<u16>, port: u16) {
        if port != 0 && !out.contains(&port) {
            out.push(port);
        }
    }
    let mut out = Vec::new();
    if let Some(port) = held {
        push(&mut out, port);
    }
    if let Some(table) = table {
        let mut mine: Vec<u16> = table
            .iter()
            .filter(|r| r.forwards_to(ours))
            .map(|r| r.external_port)
            .collect();
        mine.sort_by_key(|&p| (p != ours.internal.port(), p));
        for port in mine {
            push(&mut out, port);
        }
    }
    push(&mut out, ours.internal.port());
    for k in 1..=DERIVED_CANDIDATES {
        push(&mut out, derived_port(ours.internal, k));
    }
    if let Some(table) = table {
        out.retain(|&p| !held_by_another(table, ours, p));
    }
    out
}

/// The `k`th port derived from this address: the same on every run (FNV-1a).
fn derived_port(internal: SocketAddr, k: u8) -> u16 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut eat = |b: u8| {
        hash ^= u32::from(b);
        hash = hash.wrapping_mul(0x0100_0193);
    };
    match internal.ip() {
        IpAddr::V4(ip) => ip.octets().into_iter().for_each(&mut eat),
        IpAddr::V6(ip) => ip.octets().into_iter().for_each(&mut eat),
    }
    internal.port().to_be_bytes().into_iter().for_each(&mut eat);
    eat(k);
    DERIVED_BASE + (hash % DERIVED_SPAN) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    const TCP: PortMappingProtocol = PortMappingProtocol::TCP;
    const UDP: PortMappingProtocol = PortMappingProtocol::UDP;

    /// This PC on 2026-10-24: the LAN address and WSL's internal one.
    const OWN: [IpAddr; 2] = [
        IpAddr::V4(std::net::Ipv4Addr::new(192, 168, 1, 52)),
        IpAddr::V4(std::net::Ipv4Addr::new(10, 255, 255, 254)),
    ];

    fn me() -> SocketAddr {
        "192.168.1.52:8810".parse().unwrap()
    }

    fn ours(protocol: PortMappingProtocol, internal: SocketAddr) -> Ours<'static> {
        Ours {
            protocol,
            internal,
            own_ips: &OWN,
        }
    }

    fn row(
        external_port: u16,
        protocol: PortMappingProtocol,
        client: &str,
        internal_port: u16,
    ) -> Row {
        Row {
            external_port,
            protocol,
            internal_client: client.into(),
            internal_port,
        }
    }

    #[test]
    fn a_free_router_gets_the_internal_port_first() {
        let c = candidates(ours(TCP, me()), None, Some(&[]));
        assert_eq!(c[0], 8810);
        assert_eq!(c.len(), 1 + DERIVED_CANDIDATES as usize);
    }

    #[test]
    fn a_port_another_computer_holds_is_never_asked_for() {
        // The 2026-10-24 table: the LAN box held 8810/TCP and 8800/UDP.
        let table = [
            row(8810, TCP, "192.168.1.200", 8810),
            row(8800, UDP, "192.168.1.200", 8800),
        ];
        let c = candidates(ours(TCP, me()), None, Some(&table));
        assert!(!c.contains(&8810), "{c:?}");
        assert!(!c.is_empty());
        assert!(c.iter().all(|&p| (40_000..50_000).contains(&p)), "{c:?}");
        // The same port number on the OTHER protocol is someone else's business.
        let c = candidates(ours(UDP, me()), None, Some(&table));
        assert_eq!(c[0], 8810);
    }

    #[test]
    fn a_renewal_yields_a_port_another_computer_took() {
        // We held 8810; a router that replaces gave it to the LAN box since.
        let table = [row(8810, TCP, "192.168.1.200", 8810)];
        let c = candidates(ours(TCP, me()), Some(8810), Some(&table));
        assert!(!c.contains(&8810), "{c:?}");
        // Still ours: kept, first.
        let table = [row(8810, TCP, "192.168.1.52", 8810)];
        assert_eq!(
            candidates(ours(TCP, me()), Some(8810), Some(&table))[0],
            8810
        );
    }

    #[test]
    fn a_held_port_comes_first_and_a_row_already_ours_is_reused() {
        let moved = derived_port(me(), 3);
        assert_eq!(
            candidates(ours(TCP, me()), Some(moved), Some(&[]))[0],
            moved
        );
        // A killed run left a row forwarding to this address: reuse it.
        let table = [
            row(8810, TCP, "192.168.1.200", 8810),
            row(45_123, TCP, "192.168.1.52", 8810),
        ];
        assert_eq!(candidates(ours(TCP, me()), None, Some(&table))[0], 45_123);
        // A row to this computer but ANOTHER internal port is another node here.
        let table = [row(8810, TCP, "192.168.1.52", 8811)];
        assert!(!candidates(ours(TCP, me()), None, Some(&table)).contains(&8810));
    }

    #[test]
    fn a_row_left_at_another_of_this_computers_addresses_is_taken_back() {
        // 0.3.249 mapped WSL's internal address too and the router kept the row
        // (01:24Z 10-24: UDP 8800 -> 10.255.255.254:8800, nobody reachable there).
        let table = [row(8800, UDP, "10.255.255.254", 8800)];
        let quic: SocketAddr = "192.168.1.52:8800".parse().unwrap();
        assert_eq!(candidates(ours(UDP, quic), None, Some(&table))[0], 8800);
        assert!(!held_by_another(&table, ours(UDP, quic), 8800));
        // Without this computer's other addresses it reads as someone else's.
        let blind = Ours {
            own_ips: &[],
            ..ours(UDP, quic)
        };
        assert!(held_by_another(&table, blind, 8800));
    }

    #[test]
    fn a_router_that_will_not_list_falls_back_to_upstreams_order() {
        let c = candidates(ours(TCP, me()), None, None);
        assert_eq!(c[0], 8810);
        assert_eq!(
            candidates(ours(TCP, me()), Some(41_000), None)[..2],
            [41_000, 8810]
        );
    }

    /// Feed `answers` to a read; return the rows and how many indices were asked.
    fn read(answers: Vec<Answer>, found: impl Fn(&Row) -> bool) -> (Option<Vec<Row>>, usize) {
        let mut answers = answers.into_iter();
        let mut read = TableRead::default();
        let mut asked = 0;
        while read.next().is_some() {
            asked += 1;
            read.answer(answers.next().unwrap_or(Answer::NoEntry), &found);
        }
        (read.rows(), asked)
    }

    #[test]
    fn an_empty_slot_in_the_routers_table_does_not_end_it() {
        // The 2026-10-24 router at 01:52Z: a stopping node had emptied slots 0
        // and 5; six rows remained, the other computer's among them.
        let answers = vec![
            Answer::NoEntry,
            Answer::Row(row(8809, UDP, "192.168.1.52", 8809)),
            Answer::Row(row(8810, TCP, "192.168.1.200", 8810)),
            Answer::Row(row(8821, UDP, "192.168.1.52", 8821)),
            Answer::Row(row(8823, UDP, "192.168.1.52", 8823)),
            Answer::NoEntry,
            Answer::Row(row(47_681, UDP, "192.168.1.52", 8800)),
        ];
        let (rows, _) = read(answers, |_| false);
        let rows = rows.unwrap();
        assert_eq!(rows.len(), 5);
        assert!(held_by_another(&rows, ours(TCP, me()), 8810));
        // And this computer's own row past the second gap is found to reuse.
        let quic: SocketAddr = "192.168.1.52:8800".parse().unwrap();
        assert_eq!(candidates(ours(UDP, quic), None, Some(&rows))[0], 47_681);
    }

    #[test]
    fn a_read_ends_after_enough_empty_answers_or_at_the_row_it_wants() {
        let (rows, asked) = read(
            vec![Answer::Row(row(8810, TCP, "192.168.1.52", 8810))],
            |_| false,
        );
        assert_eq!(rows.unwrap().len(), 1);
        assert_eq!(asked, 1 + GAPS_THAT_END_THE_TABLE as usize);
        // A removal stops at its own row.
        let (rows, asked) = read(
            vec![
                Answer::Row(row(8810, TCP, "192.168.1.52", 8810)),
                Answer::Row(row(8800, UDP, "192.168.1.52", 8800)),
            ],
            |r| r.external_port == 8810,
        );
        assert_eq!((rows.unwrap().len(), asked), (1, 1));
    }

    #[test]
    fn a_router_that_will_not_list_is_read_as_unknown() {
        assert!(read(vec![Answer::Error], |_| false).0.is_none());
        // An error further on is a gap, not the router refusing.
        let (rows, _) = read(
            vec![
                Answer::Row(row(8810, TCP, "192.168.1.52", 8810)),
                Answer::Error,
                Answer::Row(row(8800, UDP, "192.168.1.52", 8800)),
            ],
            |_| false,
        );
        assert_eq!(rows.unwrap().len(), 2);
    }

    #[test]
    fn derived_ports_are_stable_and_differ_by_address() {
        assert_eq!(derived_port(me(), 1), derived_port(me(), 1));
        let other: SocketAddr = "192.168.1.200:8810".parse().unwrap();
        assert_ne!(
            (1..=DERIVED_CANDIDATES)
                .map(|k| derived_port(me(), k))
                .collect::<Vec<_>>(),
            (1..=DERIVED_CANDIDATES)
                .map(|k| derived_port(other, k))
                .collect::<Vec<_>>()
        );
    }
}
