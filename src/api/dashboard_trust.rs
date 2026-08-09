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
/// NAT, so it is not by itself proof of a tailnet, on either side of the
/// connection. That is why every caller pairs this with [`node_is_on_overlay`],
/// which requires Tailscale-SPECIFIC evidence that this node is a member.
/// "We also hold a 100.x address" is not such evidence — see that function.
/// The IPv6 half is Tailscale-specific and is the strong signal.
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

/// Is this node genuinely a member of a Tailscale-style overlay?
///
/// **Only Tailscale-specific evidence counts.** An earlier version accepted any
/// address of ours inside `100.64.0.0/10`, reasoning that an ISP's CGNAT could
/// not give us one. An external security review showed it can, and by a path
/// the reasoning never considered: `listen_multiaddrs` is the union of
/// `swarm.listeners()` — every locally BOUND interface address, since the
/// daemon binds `0.0.0.0` — with confirmed external addresses, and
/// `addr_is_remotely_reachable` deliberately keeps CGNAT. So a host whose own
/// interface sits in that block for an unrelated reason (a cellular carrier
/// numbering the device directly, an ISP numbering a customer LAN, a
/// coincidental VPN or container interface) declared itself "on the overlay"
/// having never joined a tailnet — and would then hand its API key to any
/// browser sharing that address space, which for an ISP pool means unrelated
/// customers.
///
/// The same file already had the correct instinct one function away:
/// `publicly_reachable` is computed from confirmed external addresses and
/// *never* a bound listener, because "binding a socket says nothing about
/// whether anyone outside can reach it". Membership of an overlay is the same
/// kind of claim, and needs the same kind of evidence.
///
/// Two signals, either sufficient, both Tailscale-specific:
///
///  * an address of ours inside `fd7a:115c:a1e0::/48` — Tailscale's own ULA
///    prefix, which is not shared space and cannot be handed out by an ISP;
///  * a network interface named `tailscale*` (Linux, where interface names are
///    enumerable without a dependency).
///
/// Deliberately fails closed: a tailnet node with IPv6 disabled on a platform
/// whose interfaces we cannot enumerate simply is not auto-trusted, and uses
/// the paste-the-key path like any other untrusted origin. Wrongly withholding
/// a key costs one paste; wrongly handing it out costs the node.
pub fn node_is_on_overlay(state: &SharedState) -> bool {
    let holds_tailscale_ula = state
        .listen_multiaddrs
        .load()
        .iter()
        .any(|addr| multiaddr_has_tailscale_v6(addr));
    holds_tailscale_ula || tailscale_interface_present()
}

/// Is there a Tailscale interface on this host?
///
/// Linux only, by reading `/sys/class/net` — no new dependency, and the answer
/// is a directory listing rather than anything that touches the network. Other
/// platforms return false and fall back to the ULA signal above; `utun` on
/// macOS is used by every VPN, so matching it would reintroduce exactly the
/// coincidence this function exists to rule out.
fn tailscale_interface_present() -> bool {
    #[cfg(target_os = "linux")]
    {
        let Ok(entries) = std::fs::read_dir("/sys/class/net") else {
            return false;
        };
        entries.filter_map(Result::ok).any(|e| {
            e.file_name()
                .to_str()
                .is_some_and(|n| n.starts_with("tailscale"))
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

/// Does this multiaddr carry an address in Tailscale's own IPv6 ULA prefix?
///
/// Deliberately IPv6-only. The v4 CGNAT range is shared address space and
/// proves nothing about tailnet membership — treating it as proof is the defect
/// this function was narrowed to fix. Parsing textually keeps the helper free
/// of a libp2p dependency so it stays unit-testable in isolation.
fn multiaddr_has_tailscale_v6(addr: &str) -> bool {
    let mut parts = addr.split('/');
    while let Some(seg) = parts.next() {
        if seg != "ip6" {
            continue;
        }
        let Some(literal) = parts.next() else { break };
        if let Ok(v6) = literal.parse::<Ipv6Addr>() {
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
pub async fn classify(state: &SharedState, ip: IpAddr) -> DashboardTrust {
    if ip.is_loopback() {
        return DashboardTrust::Loopback;
    }
    // Ask Tailscale who this is, rather than inferring it from the address.
    //
    // `whois` is authoritative in a way no address test can be: it answers for
    // BOTH sides at once — a daemon that is not there means we are not on a
    // tailnet, and an address it does not recognise means the caller is not on
    // ours. Only `Member` grants trust; `Unavailable` means we could not ask
    // and must not be read as a yes.
    //
    // The address test remains as a fallback for hosts where the socket is not
    // readable (a sandboxed service cannot open it), and is narrowed to
    // Tailscale-specific evidence for the reason described on
    // `node_is_on_overlay`.
    if is_overlay_ip(ip) && state.config.api.dashboard_trust_overlay {
        match crate::api::tailscale::whois(ip).await {
            crate::api::tailscale::WhoIs::Member => return DashboardTrust::Overlay,
            crate::api::tailscale::WhoIs::NotAMember => {
                // A definitive no. Do not fall through to the weaker test —
                // Tailscale has told us this address is not one of ours.
            }
            crate::api::tailscale::WhoIs::Unavailable => {
                if node_is_on_overlay(state) {
                    return DashboardTrust::Overlay;
                }
            }
        }
    }
    // Live config, NOT the boot snapshot: the user flips this precisely when
    // their dashboard is unreachable, so it has to apply without a restart of
    // the node they cannot reach. See SharedState::cfg.
    let trust_lan = state.cfg().api.dashboard_trust_lan;
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
    #[tokio::test]
    async fn tailnet_browser_is_trusted_when_this_node_is_on_the_tailnet() {
        let state = test_state(crate::config::Config::default());
        let peer = v4("100.101.102.103");

        // Before any Tailscale-specific evidence, a 100.x source is just
        // shared CGNAT space and proves nothing.
        assert_eq!(classify(&state, peer).await, DashboardTrust::Untrusted);

        // Tailscale gives every node an address in its own IPv6 ULA prefix.
        // That is not shared space, so holding one is proof of membership.
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip4/100.64.0.7/tcp/8810".into(),
            "/ip6/fd7a:115c:a1e0::1234/tcp/8810".into(),
        ]));
        assert_eq!(classify(&state, peer).await, DashboardTrust::Overlay);
    }

    /// Both CGNAT tests below assert what happens with NO Tailscale-specific
    /// evidence, so they are meaningless on a machine that genuinely runs
    /// Tailscale — the interface probe would (correctly) return true. Skip
    /// rather than fail: a contributor on a tailnet should not see a red test
    /// for being on a tailnet.
    fn skip_if_host_runs_tailscale() -> bool {
        if tailscale_interface_present() {
            eprintln!("skipped: this host has a tailscale interface");
            return true;
        }
        false
    }

    /// The finding from an external security review, 2026-07-28.
    ///
    /// `listen_multiaddrs` includes every locally BOUND address (the daemon
    /// binds 0.0.0.0), and CGNAT is deliberately kept as "remotely reachable".
    /// So a host numbered inside 100.64.0.0/10 for a reason that has nothing to
    /// do with Tailscale — a cellular carrier addressing the device directly,
    /// an ISP numbering a customer LAN, a coincidental VPN or container
    /// interface — used to classify itself as being on the overlay, and would
    /// then hand its API key to any browser in that same space. On an ISP pool
    /// that means unrelated customers of that ISP.
    #[tokio::test]
    async fn our_own_cgnat_address_is_not_proof_of_a_tailnet() {
        if skip_if_host_runs_tailscale() {
            return;
        }
        let state = test_state(crate::config::Config::default());
        // Every one of these is a plausible non-Tailscale CGNAT interface.
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip4/100.64.0.7/tcp/8810".into(),
            "/ip4/100.96.13.2/udp/8800/quic-v1".into(),
            "/ip4/192.168.1.53/tcp/8810".into(),
        ]));
        assert!(
            !state
                .listen_multiaddrs
                .load()
                .iter()
                .any(|a| multiaddr_has_tailscale_v6(a)),
            "no Tailscale-specific evidence is present"
        );
        assert_eq!(
            classify(&state, v4("100.101.102.103")).await,
            DashboardTrust::Untrusted,
            "a CGNAT neighbour must not be handed the API key on this evidence"
        );
    }

    /// A non-Tailscale IPv6 ULA is not proof either — only Tailscale's prefix.
    #[tokio::test]
    async fn a_generic_ipv6_ula_is_not_proof_of_a_tailnet() {
        if skip_if_host_runs_tailscale() {
            return;
        }
        let state = test_state(crate::config::Config::default());
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip6/fd00:dead:beef::1/tcp/8810".into()
        ]));
        assert_eq!(
            classify(&state, v4("100.101.102.103")).await,
            DashboardTrust::Untrusted
        );
    }

    /// An ISP that puts its customers behind carrier-grade NAT hands out
    /// addresses from the same 100.64.0.0/10 block Tailscale uses. A node that
    /// is NOT on a tailnet must not treat such a neighbour as trusted — which
    /// is why membership is proven from our own addresses, not the peer's.
    #[tokio::test]
    async fn cgnat_neighbour_is_untrusted_when_we_are_not_on_a_tailnet() {
        if skip_if_host_runs_tailscale() {
            return;
        }
        let state = test_state(crate::config::Config::default());
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip4/192.168.1.53/tcp/8810".into(),
            "/ip4/203.0.113.9/tcp/8810".into(),
        ]));
        assert_eq!(
            classify(&state, v4("100.101.102.103")).await,
            DashboardTrust::Untrusted
        );
    }

    /// The subnet-router case: Tailscale masquerades by default, so traffic
    /// from the tailnet arrives from the router's own private address. Nothing
    /// distinguishes it from any other LAN client, so it takes the explicit
    /// opt-in — and that opt-in must apply without a restart.
    #[tokio::test]
    async fn lan_browser_is_trusted_only_after_the_opt_in() {
        let state = test_state(crate::config::Config::default());
        let router = v4("192.168.1.10");
        assert_eq!(classify(&state, router).await, DashboardTrust::Untrusted);

        let mut opted_in = (**state.cfg()).clone();
        opted_in.api.dashboard_trust_lan = true;
        state.apply_live_config(opted_in);
        assert_eq!(classify(&state, router).await, DashboardTrust::LocalNetwork);

        // A public address is never admitted by the LAN opt-in.
        assert_eq!(
            classify(&state, v4("8.8.8.8")).await,
            DashboardTrust::Untrusted
        );
    }

    #[tokio::test]
    async fn loopback_is_always_trusted() {
        let state = test_state(crate::config::Config::default());
        assert_eq!(
            classify(&state, v4("127.0.0.1")).await,
            DashboardTrust::Loopback
        );
        assert_eq!(classify(&state, v6("::1")).await, DashboardTrust::Loopback);
    }

    /// Turning the overlay off is the escape hatch for someone sharing a
    /// tailnet with people they would not give admin to.
    #[tokio::test]
    async fn overlay_trust_can_be_disabled() {
        let mut config = crate::config::Config::default();
        config.api.dashboard_trust_overlay = false;
        let state = test_state(config);
        state
            .listen_multiaddrs
            .store(std::sync::Arc::new(vec!["/ip4/100.64.0.7/tcp/8810".into()]));
        assert_eq!(
            classify(&state, v4("100.101.102.103")).await,
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
    fn only_the_tailscale_ula_is_found_in_listen_multiaddrs() {
        assert!(multiaddr_has_tailscale_v6(
            "/ip6/fd7a:115c:a1e0::1/udp/8800/quic-v1"
        ));
        // v4 CGNAT is shared space — never evidence, however it is written.
        assert!(!multiaddr_has_tailscale_v6("/ip4/100.101.102.103/tcp/8810"));
        assert!(!multiaddr_has_tailscale_v6("/ip6/fd00::1/tcp/8810"));
        assert!(!multiaddr_has_tailscale_v6("/ip4/192.168.1.53/tcp/8810"));
        // A p2p-only address carries no IP and must not panic or match.
        assert!(!multiaddr_has_tailscale_v6("/p2p/12D3KooWFake"));
        // Truncated input: `/ip6` with no literal following.
        assert!(!multiaddr_has_tailscale_v6("/ip6"));
    }
}
