//! Who may be handed the dashboard's API key automatically.
//!
//! The dashboard authenticates every admin call with a Bearer token it does
//! not ask the user for — it fetches it from `GET /api/admin/api-key` on page
//! load. That handout is the only thing standing between a browser and admin
//! access, so it is gated on where the request came from.
//!
//! For a long time the gate was exactly `addr.ip().is_loopback()`. That reads
//! like "the user is sitting at this machine", but it actually means "the last
//! TCP hop originated inside this daemon's network namespace" — which is a
//! different, and much narrower, claim:
//!
//!   * A reverse proxy on the same host (a plain `tailscale serve`) hands the
//!     key to a fully remote phone, because the proxy dials us over loopback.
//!   * A container publish, a NAT, or a Tailscale *subnet router* never
//!     satisfies it — not even from the host's own `localhost`, because the
//!     packet crosses into the container's namespace and is masqueraded on the
//!     way. Tailscale subnet routers SNAT by default.
//!
//! So the loopback test both over- and under-approximates the thing we care
//! about. This module states the real question — *is the network this request
//! arrived over one the operator has vouched for?* — and answers it in one
//! place, so no call site re-derives it.
//!
//! Trust here only ever unlocks the *automatic* handout. It is deliberately
//! not a way to skip authentication: an untrusted origin can still use the
//! dashboard by pasting the API key, and every admin endpoint still demands a
//! Bearer token either way.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use crate::daemon::state::SharedState;

/// Why a request's source address is (or isn't) allowed the key handout.
///
/// Carried to the frontend so the dashboard can explain a refusal in terms of
/// the user's own network rather than a bare 401.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardTrust {
    /// Same network namespace as the daemon.
    Loopback,
    /// A Tailscale-style overlay address, and we are on that overlay too.
    Overlay,
    /// A private/LAN address, with `api.dashboard_trust_lan` enabled.
    LocalNetwork,
    /// Anything else — the browser must supply the key itself.
    Untrusted,
}

impl DashboardTrust {
    pub fn is_trusted(self) -> bool {
        !matches!(self, DashboardTrust::Untrusted)
    }

    /// Stable machine-readable tag, embedded in the dashboard HTML.
    pub fn as_str(self) -> &'static str {
        match self {
            DashboardTrust::Loopback => "loopback",
            DashboardTrust::Overlay => "overlay",
            DashboardTrust::LocalNetwork => "local-network",
            DashboardTrust::Untrusted => "untrusted",
        }
    }
}

/// Tailscale's address space: RFC 6598 CGNAT `100.64.0.0/10` for IPv4 and the
/// `fd7a:115c:a1e0::/48` unique-local prefix for IPv6.
///
/// The IPv4 half is *shared* address space — real ISPs use it for carrier-grade
/// NAT, so it is not by itself proof of a tailnet. That is why every caller
/// pairs this with [`node_is_on_overlay`]: we only extend trust across this
/// range when this node holds an address in it too, which an ISP's CGNAT
/// segment does not give us. The IPv6 half is Tailscale-specific.
pub fn is_overlay_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_cgnat_v4(v4),
        IpAddr::V6(v6) => is_tailscale_v6(v6),
    }
}

fn is_cgnat_v4(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (64..128).contains(&o[1])
}

fn is_tailscale_v6(ip: Ipv6Addr) -> bool {
    let s = ip.segments();
    s[0] == 0xfd7a && s[1] == 0x115c && s[2] == 0xa1e0
}

/// A private address on some local network: RFC1918, IPv6 unique-local, or
/// link-local. Excludes the overlay ranges, which are classified first and
/// carry a stronger claim.
pub fn is_local_network_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            let s = v6.segments();
            // fc00::/7 unique-local, fe80::/10 link-local.
            ((s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80) && !is_tailscale_v6(v6)
        }
    }
}

/// Does this node itself hold an overlay address?
///
/// Read from `state.listen_multiaddrs`, which the NetworkManager keeps as the
/// union of bound sockets and confirmed external addresses. The API server
/// binds `0.0.0.0`, so a machine that joined a tailnet has its overlay address
/// enumerated there as a listener without us needing to walk interfaces
/// ourselves or take a new dependency.
pub fn node_is_on_overlay(state: &SharedState) -> bool {
    state
        .listen_multiaddrs
        .load()
        .iter()
        .any(|addr| multiaddr_overlay_ip(addr))
}

/// Pull an IP literal out of a multiaddr string (`/ip4/100.64.0.5/tcp/8810/...`)
/// and test it. Parsing textually keeps this helper free of a libp2p dependency
/// so it stays unit-testable in isolation.
fn multiaddr_overlay_ip(addr: &str) -> bool {
    let mut parts = addr.split('/');
    while let Some(seg) = parts.next() {
        let is_v4 = seg == "ip4";
        let is_v6 = seg == "ip6";
        if !is_v4 && !is_v6 {
            continue;
        }
        let Some(literal) = parts.next() else { break };
        if is_v4 {
            if let Ok(v4) = literal.parse::<Ipv4Addr>() {
                if is_cgnat_v4(v4) {
                    return true;
                }
            }
        } else if let Ok(v6) = literal.parse::<Ipv6Addr>() {
            if is_tailscale_v6(v6) {
                return true;
            }
        }
    }
    false
}

/// Classify a request's source address against this node's configuration.
///
/// Order matters: loopback first (cheapest and always trusted), then the
/// overlay (an authenticated network), then the LAN (an opt-in). A Tailscale
/// address is reported as `Overlay` even when LAN trust is what would also have
/// admitted it, so the dashboard explains the specific reason.
pub fn classify(state: &SharedState, ip: IpAddr) -> DashboardTrust {
    if ip.is_loopback() {
        return DashboardTrust::Loopback;
    }
    if is_overlay_ip(ip) && state.config.api.dashboard_trust_overlay && node_is_on_overlay(state) {
        return DashboardTrust::Overlay;
    }
    // Runtime atomic, NOT `config.api.dashboard_trust_lan`: the user flips this
    // precisely when their dashboard is unreachable, so it has to apply without
    // a restart of the node they cannot reach. See SharedState::dashboard_trust_lan.
    let trust_lan = state
        .dashboard_trust_lan
        .load(std::sync::atomic::Ordering::Relaxed);
    if trust_lan && (is_local_network_ip(ip) || is_overlay_ip(ip)) {
        return DashboardTrust::LocalNetwork;
    }
    DashboardTrust::Untrusted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(s: &str) -> IpAddr {
        IpAddr::V4(s.parse().unwrap())
    }
    fn v6(s: &str) -> IpAddr {
        IpAddr::V6(s.parse().unwrap())
    }

    #[test]
    fn tailscale_ranges_are_recognised() {
        // Boundaries of 100.64.0.0/10 — 100.64.0.0 through 100.127.255.255.
        assert!(is_overlay_ip(v4("100.64.0.0")));
        assert!(is_overlay_ip(v4("100.101.102.103")));
        assert!(is_overlay_ip(v4("100.127.255.255")));
        // Just outside on either side.
        assert!(!is_overlay_ip(v4("100.63.255.255")));
        assert!(!is_overlay_ip(v4("100.128.0.0")));
        // Tailscale's ULA prefix, and a non-Tailscale ULA.
        assert!(is_overlay_ip(v6("fd7a:115c:a1e0::1")));
        assert!(!is_overlay_ip(v6("fd00::1")));
    }

    #[test]
    fn lan_ranges_exclude_the_overlay() {
        assert!(is_local_network_ip(v4("192.168.1.10")));
        assert!(is_local_network_ip(v4("10.0.0.5")));
        assert!(is_local_network_ip(v4("172.16.4.4")));
        assert!(is_local_network_ip(v6("fd00::1")));
        // Tailscale's own ULA must not be reported as a plain LAN address —
        // it is classified as the (stronger) overlay case instead.
        assert!(!is_local_network_ip(v6("fd7a:115c:a1e0::1")));
        assert!(!is_local_network_ip(v4("8.8.8.8")));
        // CGNAT is deliberately NOT a LAN range here; it is the overlay range.
        assert!(!is_local_network_ip(v4("100.64.0.1")));
    }

    fn test_state(config: crate::config::Config) -> std::sync::Arc<SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = SharedState::new(config, identity, db, executor, None);
        state
    }

    /// The whole point of the feature: a node that has joined a tailnet serves
    /// a working dashboard to the tailnet without any configuration. We
    /// document running nodes over Tailscale, so this is the advertised path.
    #[test]
    fn tailnet_browser_is_trusted_when_this_node_is_on_the_tailnet() {
        let state = test_state(crate::config::Config::default());
        let peer = v4("100.101.102.103");

        // Before the swarm reports an overlay address of our own, a 100.x
        // source is just shared CGNAT space and proves nothing.
        assert_eq!(classify(&state, peer), DashboardTrust::Untrusted);

        state
            .listen_multiaddrs
            .store(std::sync::Arc::new(vec!["/ip4/100.64.0.7/tcp/8810".into()]));
        assert_eq!(classify(&state, peer), DashboardTrust::Overlay);
    }

    /// An ISP that puts its customers behind carrier-grade NAT hands out
    /// addresses from the same 100.64.0.0/10 block Tailscale uses. A node that
    /// is NOT on a tailnet must not treat such a neighbour as trusted — which
    /// is why membership is proven from our own addresses, not the peer's.
    #[test]
    fn cgnat_neighbour_is_untrusted_when_we_are_not_on_a_tailnet() {
        let state = test_state(crate::config::Config::default());
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip4/192.168.1.53/tcp/8810".into(),
            "/ip4/203.0.113.9/tcp/8810".into(),
        ]));
        assert_eq!(
            classify(&state, v4("100.101.102.103")),
            DashboardTrust::Untrusted
        );
    }

    /// The subnet-router case: Tailscale masquerades by default, so traffic
    /// from the tailnet arrives from the router's own private address. Nothing
    /// distinguishes it from any other LAN client, so it takes the explicit
    /// opt-in — and that opt-in must apply without a restart.
    #[test]
    fn lan_browser_is_trusted_only_after_the_opt_in() {
        let state = test_state(crate::config::Config::default());
        let router = v4("192.168.1.10");
        assert_eq!(classify(&state, router), DashboardTrust::Untrusted);

        state
            .dashboard_trust_lan
            .store(true, std::sync::atomic::Ordering::Release);
        assert_eq!(classify(&state, router), DashboardTrust::LocalNetwork);

        // A public address is never admitted by the LAN opt-in.
        assert_eq!(classify(&state, v4("8.8.8.8")), DashboardTrust::Untrusted);
    }

    #[test]
    fn loopback_is_always_trusted() {
        let state = test_state(crate::config::Config::default());
        assert_eq!(classify(&state, v4("127.0.0.1")), DashboardTrust::Loopback);
        assert_eq!(classify(&state, v6("::1")), DashboardTrust::Loopback);
    }

    /// Turning the overlay off is the escape hatch for someone sharing a
    /// tailnet with people they would not give admin to.
    #[test]
    fn overlay_trust_can_be_disabled() {
        let mut config = crate::config::Config::default();
        config.api.dashboard_trust_overlay = false;
        let state = test_state(config);
        state
            .listen_multiaddrs
            .store(std::sync::Arc::new(vec!["/ip4/100.64.0.7/tcp/8810".into()]));
        assert_eq!(
            classify(&state, v4("100.101.102.103")),
            DashboardTrust::Untrusted
        );
    }

    /// `#[derive(Default)]` ignores `#[serde(default = "...")]`, and a fresh
    /// install with no config file goes through `Default` — which shipped the
    /// overlay default as `false` and then wrote that back to config.toml.
    #[test]
    fn fresh_install_defaults_trust_the_overlay_but_not_the_lan() {
        let api = crate::config::Config::default().api;
        assert!(api.dashboard_trust_overlay);
        assert!(!api.dashboard_trust_lan);
    }

    #[test]
    fn overlay_address_is_found_in_listen_multiaddrs() {
        assert!(multiaddr_overlay_ip("/ip4/100.101.102.103/tcp/8810"));
        assert!(multiaddr_overlay_ip(
            "/ip6/fd7a:115c:a1e0::1/udp/8800/quic-v1"
        ));
        assert!(!multiaddr_overlay_ip("/ip4/192.168.1.53/tcp/8810"));
        assert!(!multiaddr_overlay_ip("/ip4/127.0.0.1/tcp/8810"));
        // A p2p-only address carries no IP and must not panic or match.
        assert!(!multiaddr_overlay_ip("/p2p/12D3KooWFake"));
        // Truncated input: `/ip4` with no literal following.
        assert!(!multiaddr_overlay_ip("/ip4"));
    }
}
