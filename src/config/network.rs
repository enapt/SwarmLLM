//! Network/transport configuration.
//!
//! Hosts `NetworkConfig` (peer-exchange, relay, mDNS, NAT/DCUtR, gossip
//! ID, encryption, tensor + prefix-KV compression, listen address, QUIC),
//! its `Default` impl, and the network-only default helpers (max_peers,
//! relay capacity, compression level/threshold, listen address). Also
//! exposes `is_wsl2` — used by Config::load_or_create to apply
//! WSL2-safe network overrides.

use super::default_true;
use serde::{Deserialize, Serialize};
use swarmllm_types::ContributionMode;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default = "default_bootstrap_peers")]
    pub bootstrap_peers: Vec<String>,
    /// Genuinely run with NO bootstrap peers, rather than falling back to the
    /// built-in anchors.
    ///
    /// This exists because `bootstrap_peers = []` cannot mean "no peers": every
    /// config written before 2026-07-21 has that line, put there by the daemon
    /// itself when an empty list WAS the default. Those nodes are not opting
    /// out of anything — they were saved that way — and after the built-in
    /// anchor landed they became permanently unable to find the swarm, with an
    /// empty peer list and nothing explaining why. An empty list is therefore
    /// treated as "not configured" and falls back to the built-ins; say so here
    /// to actually mean it.
    ///
    /// Implied by `node.anchor_mode` — an anchor IS the bootstrap and must not
    /// dial itself. Set it for a private or air-gapped swarm that must never
    /// contact the public anchors.
    #[serde(default)]
    pub disable_default_bootstrap: bool,
    #[serde(default = "default_true")]
    pub peer_exchange: bool,
    #[serde(default = "default_true")]
    pub enable_relay: bool,
    #[serde(default = "default_true")]
    pub enable_relay_client: bool,
    /// Ceiling on simultaneously established peer connections.
    ///
    /// `None` (the default) resolves from `node.contribution` — see
    /// [`NetworkConfig::effective_max_connections`]. An explicit value always
    /// wins, in either direction.
    #[serde(default)]
    pub max_peers: Option<u32>,
    /// Maximum duration for a single relay circuit in seconds.
    #[serde(default = "default_relay_circuit_duration")]
    pub relay_max_circuit_duration_secs: u64,
    /// Maximum number of relay circuits this node will serve simultaneously.
    #[serde(default = "default_relay_max_circuits")]
    pub relay_max_circuits: usize,
    /// Automatically activate relay listener when NAT is detected as Private.
    #[serde(default = "default_true")]
    pub auto_relay: bool,
    /// NETWORKING_PLAN Phase 1 — forward *inference* messages between two peers
    /// that cannot reach each other directly (application-level relay). This is
    /// distinct from libp2p circuit-relay (`enable_relay`): that carries the
    /// connection + gossip, this carries the end-to-end-sealed inference payload
    /// as a dumb pipe so two NAT'd nodes can actually complete a request. A
    /// forwarding node advertises `relay_capable` so peers only route through it
    /// deliberately. Auto-enabled in `--anchor` mode; a publicly-reachable
    /// non-anchor node can opt in to donate relay capacity. Default off for
    /// ordinary NAT'd nodes (they can't forward anyway).
    #[serde(default)]
    pub relay_forwarding: bool,
    /// NETWORKING_PLAN Phase 3 — automatically donate relay capacity once this
    /// node is confirmed reachable from the open internet (UPnP-mapped,
    /// AutoNAT-confirmed, or a declared external address).
    ///
    /// Phase 3 says "any public node can opt in as a relay", but `relay_forwarding`
    /// is a flag nothing ever sets, so in practice the swarm's whole relay
    /// capacity was the handful of `--anchor` nodes — a single point of failure
    /// and a throughput ceiling for every NAT'd pair. Defaulting this on makes
    /// relay capacity grow with the swarm's public membership, which is the
    /// property Phase 3 depends on.
    ///
    /// Only ever true for genuinely public nodes: a NAT'd node can't forward,
    /// and reachability *through a relay circuit* does not count. Set to false
    /// to keep a public node from donating upload bandwidth.
    #[serde(default = "default_true")]
    pub relay_forwarding_auto: bool,
    /// Maximum simultaneous libp2p connections to a single peer.
    ///
    /// **Must be at least 2 for NAT hole punching to work.** DCUtR upgrades a
    /// relayed connection by dialling a direct one *while the relayed one is
    /// still open*, so a value of 1 causes `connection_limits` to deny the
    /// upgrade and the node never escapes the relay.
    ///
    /// This is exposed as an escape hatch, not a tuning knob. It was `1` before
    /// the 2026-07-25 networking audit, to work around upstream
    /// request_response round-robining across a half-open parallel connection on
    /// multi-interface hosts. Two changes address that properly (a vendored
    /// patch preferring direct connections, and the deterministic mDNS dialer),
    /// but if silent request drops ever reappear on a multi-NIC host, setting
    /// this to 1 restores the old behaviour without a rebuild — and confirms or
    /// rules out that cause in one step.
    #[serde(default = "default_max_connections_per_peer")]
    pub max_connections_per_peer: u32,
    /// Enable mDNS for automatic LAN peer discovery (default: true).
    #[serde(default = "default_true")]
    pub enable_mdns: bool,
    /// Gossip network ID for grouping nodes. All nodes sharing the same ID
    /// can decode each other's sealed gossip. Defaults to "swarmllm-mainnet-v1".
    /// Set to a custom value (e.g. "my-private-net") for private networks.
    #[serde(default)]
    pub gossip_network_id: Option<String>,
    /// Enable AutoNAT for NAT detection (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_autonat: bool,
    /// Enable DCUtR for hole punching (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_dcutr: bool,
    /// Enable UPnP/IGD automatic gateway port-mapping (default: true).
    /// On a home router with UPnP enabled this opens the P2P ports on the
    /// gateway and confirms the resulting public address with the swarm — the
    /// zero-config path to internet reachability for most home users. Inert
    /// (emits GatewayNotFound) on routers without UPnP. Auto-disabled on WSL2.
    #[serde(default = "default_true")]
    pub enable_upnp: bool,
    /// Manually declared external addresses for nodes that already know how
    /// they are reachable from the internet — a port-forwarded home box, a VPS,
    /// or a dynamic-DNS anchor. Each is an IP or DNS multiaddr WITHOUT the
    /// trailing `/p2p/<peer_id>` (the daemon appends its own). List both
    /// transports to advertise your readable name on TCP *and* QUIC, e.g.
    /// `["/dns4/anchor.example.net/tcp/8810", "/dns4/anchor.example.net/udp/8800/quic-v1"]`.
    /// Each is added via `Swarm::add_external_address` at startup so it flows
    /// into identify, the DHT, and every invite code this node mints. Empty
    /// (default) leaves discovery to UPnP/AutoNAT/relay + auto-advertised
    /// listeners.
    #[serde(default, alias = "external_address")]
    pub external_addresses: ExternalAddresses,
    /// Enable E2E encryption for tensor forwards and control messages (default: true).
    #[serde(default = "default_true")]
    pub enable_encryption: bool,
    /// Enable zstd compression for tensor payloads sent over the network.
    /// Only payloads larger than `tensor_compress_threshold` bytes are compressed.
    #[serde(default = "default_true")]
    pub tensor_compression: bool,
    /// Enable zstd compression for cross-node prefix-KV snapshot payloads
    /// (Item 8 wire frames, tag 0x04). Off by default — only worth flipping
    /// when WAN measurements show wire size is the binding constraint
    /// (localhost's RTT-vs-wire trade is roughly neutral). Receivers always
    /// decompress regardless of this flag, so flipping it on a single peer
    /// doesn't require a coordinated upgrade.
    #[serde(default)]
    pub prefix_kv_compression: bool,
    /// Zstd compression level (1-22, default 1 for speed). Shared between
    /// tensor and prefix-KV compression.
    #[serde(default = "default_tensor_compress_level")]
    pub tensor_compress_level: i32,
    /// Minimum payload size in bytes before compression is applied (default 1024).
    /// Shared between tensor and prefix-KV.
    #[serde(default = "default_tensor_compress_threshold")]
    pub tensor_compress_threshold: usize,
    /// IP address to bind P2P listeners on (default: "0.0.0.0" = all interfaces).
    /// Set to "127.0.0.1" on WSL2 to prevent connections via unreliable NAT adapters.
    #[serde(default = "default_listen_address")]
    pub listen_address: String,
    /// Enable QUIC transport (default: true).
    /// Disable on WSL2 to prevent QUIC connection races with TCP (QUIC handshake is faster
    /// than TCP+Noise+Yamux, causing max_established_per_peer=1 to kill the TCP connection).
    #[serde(default = "default_true")]
    pub enable_quic: bool,
}

fn default_listen_address() -> String {
    "0.0.0.0".to_string()
}

/// Default bootstrap anchor(s) — publicly-reachable seed nodes a fresh install
/// dials on startup to join the network before decentralized discovery (DHT/PEX)
/// takes over. DNS form so the entry survives a host IP change (the anchor keeps
/// its DuckDNS record pointed at its current IP). A dead anchor is harmless — the
/// dial just fails and the node falls back to mDNS/DHT. Override with an explicit
/// `bootstrap_peers` in config; to run with none at all set
/// `disable_default_bootstrap = true`, because an empty list means "not
/// configured" (see that field for why).
pub fn default_bootstrap_peers() -> Vec<String> {
    vec![
        // DNS form (primary, portable across a host IP change — requires the
        // swarm's DNS transport, wired via `.with_dns()` in the manager).
        "/dns4/swarmllm.duckdns.org/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G".to_string(),
        "/dns4/swarmllm.duckdns.org/udp/8800/quic-v1/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G".to_string(),
        // IP fallback (in case DNS resolution is unavailable). A stale IP after
        // a host move just yields one failed dial — the DNS entries still work.
        "/ip4/212.132.104.177/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G".to_string(),
    ]
}

/// Detect WSL2 by checking /proc/version for "microsoft" or "WSL".
pub(crate) fn is_wsl2() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| {
            let lower = v.to_lowercase();
            lower.contains("microsoft") || lower.contains("wsl")
        })
        .unwrap_or(false)
}

/// Parse the stdout of `wslinfo --networking-mode`.
///
/// `Some(true)` = mirrored, `Some(false)` = any other explicit mode
/// (`nat` / `none` / `virtioproxy`), `None` = no usable output so the caller
/// should fall back to the interface signal.
fn parse_wslinfo_networking_mode(stdout: &str) -> Option<bool> {
    let mode = stdout.trim();
    if mode.is_empty() {
        return None;
    }
    Some(mode.eq_ignore_ascii_case("mirrored"))
}

/// Whether WSL2 networking is running in "mirrored" mode.
///
/// In mirrored mode (Windows 11 22H2+, WSL 2.0.0+, opt-in via `.wslconfig`
/// `networkingMode=mirrored`) the VM shares the Windows host's network
/// interfaces: it gets a real LAN address, and QUIC / mDNS / UPnP / AutoNAT /
/// DCUtR all behave as on a native host. The NAT-mode "safe defaults" are then
/// actively harmful — pinning the node to loopback and disabling QUIC strands
/// an otherwise-reachable node on the relay (observed live: outbound inference
/// request_response to a public peer timed out because our only path was the
/// relay).
///
/// Detection, in order of reliability:
/// 1. `wslinfo --networking-mode` (WSL 2.0.4+) prints `mirrored` or `nat`.
/// 2. Fallback for older WSL that predates `wslinfo`: the `loopback0`
///    interface, which mirrored mode creates — the same signal Docker uses to
///    special-case mirrored mode (moby/moby#48075).
pub(crate) fn wsl_networking_is_mirrored() -> bool {
    if let Some(mirrored) = wslinfo_networking_mode(std::time::Duration::from_secs(2)) {
        return mirrored;
    }
    // wslinfo missing, errored, or hung past the timeout — fall back to the
    // interface signal.
    std::path::Path::new("/sys/class/net/loopback0").exists()
}

/// Run `wslinfo --networking-mode` with a hard timeout so it can never stall
/// startup. Config load is synchronous, so an unbounded `Command::output()`
/// would block the whole boot if `wslinfo` (a symlink to `/init`) ever hung.
/// On timeout the child is killed and `None` is returned, letting the caller
/// fall back to the `loopback0` interface check. `wslinfo` normally answers in
/// well under a millisecond, so the poll loop exits on its first iteration.
fn wslinfo_networking_mode(timeout: std::time::Duration) -> Option<bool> {
    use std::io::Read;
    let mut child = std::process::Command::new("wslinfo")
        .arg("--networking-mode")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .ok()?;
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut out = String::new();
                child.stdout.take()?.read_to_string(&mut out).ok()?;
                return parse_wslinfo_networking_mode(&out);
            }
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            Err(_) => return None,
        }
    }
}

/// Absolute ceiling on established connections, whatever the contribution mode.
///
/// This is the value the daemon enforced unconditionally before `max_peers`
/// was wired up, so no node's ceiling rises as a result of that fix. It also
/// keeps a comfortable margin under the tightest file-descriptor limit a user
/// is likely to meet — macOS still defaults `RLIMIT_NOFILE` to 256 for a
/// process started from a shell, and connections compete with shard reads and
/// the database for descriptors.
pub const MAX_ESTABLISHED_CONNECTIONS_CEILING: u32 = 500;

impl NetworkConfig {
    /// Ceiling on simultaneously established connections.
    ///
    /// **`max_peers` was inert until 2026-08-04**: it was parsed, logged at
    /// startup and shown in the dashboard, but no code ever limited anything by
    /// it. The only real cap was a hardcoded 500, so a user who set
    /// `max_peers = 20` on a constrained box got no protection and no warning.
    ///
    /// The figures are deliberately generous, because the cost of getting this
    /// wrong is asymmetric. Gossipsub bounds message amplification by the
    /// **mesh degree** (D ≈ 6–12), not by how many peers are connected, so
    /// holding 300 connections does not mean 300× the gossip traffic — the
    /// per-connection cost is mostly memory for the Noise/Yamux session. Set
    /// this too low, though, and the node cannot reach enough of the DHT to
    /// route, which partitions it from the swarm. A node that is slightly
    /// chattier than ideal is a much better failure than one that is alone.
    ///
    /// Numbers are connection counts, not distinct peers: a single peer briefly
    /// holds up to `max_connections_per_peer` while a hole punch upgrades a
    /// relayed connection to a direct one, so the steady-state peer count sits
    /// at or a little below this.
    pub fn effective_max_connections(&self, contribution: ContributionMode) -> u32 {
        match self.max_peers {
            // An explicit setting wins outright, in either direction — the same
            // rule `max_gpu_vram_mb` and `max_bandwidth_mbps` follow. Only a
            // literal 0 is refused, since it would isolate the node completely
            // rather than expressing any preference about resource use.
            Some(explicit) => explicit.max(1),
            None => match contribution {
                // Default mode. Still far more than the ~20–40 DHT contacts and
                // ~12 mesh peers a node needs to participate fully.
                ContributionMode::Minimal => 150,
                ContributionMode::Moderate => 300,
                // An explicit offer of the machine.
                ContributionMode::Maximum => MAX_ESTABLISHED_CONNECTIONS_CEILING,
            },
        }
    }
}

fn default_tensor_compress_level() -> i32 {
    1
}

fn default_tensor_compress_threshold() -> usize {
    1024
}

fn default_relay_circuit_duration() -> u64 {
    3600
}

/// 3 leaves room for a relayed + direct connection during a DCUtR upgrade, plus
/// one transport variant (a peer reachable over both TCP and QUIC), while still
/// bounding runaway parallel connections. Never set below 2 — see
/// `max_connections_per_peer`.
fn default_max_connections_per_peer() -> u32 {
    3
}

fn default_relay_max_circuits() -> usize {
    // See `network::relay::RelayServerConfig::default` — libp2p's 16 assumes a
    // 2-minute bootstrap circuit; ours are a 1-hour data path, so a slot is
    // held ~30x longer and 16 is exhausted by a small swarm.
    128
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bootstrap_peers: default_bootstrap_peers(),
            disable_default_bootstrap: false,
            peer_exchange: true,
            enable_relay: true,
            enable_relay_client: true,
            max_peers: None,
            relay_max_circuit_duration_secs: default_relay_circuit_duration(),
            relay_max_circuits: default_relay_max_circuits(),
            auto_relay: true,
            relay_forwarding: false,
            relay_forwarding_auto: true,
            max_connections_per_peer: default_max_connections_per_peer(),
            enable_mdns: true,
            enable_autonat: true,
            enable_dcutr: true,
            enable_upnp: true,
            external_addresses: ExternalAddresses::default(),
            enable_encryption: true,
            gossip_network_id: None,
            tensor_compression: true,
            prefix_kv_compression: false,
            tensor_compress_level: default_tensor_compress_level(),
            tensor_compress_threshold: default_tensor_compress_threshold(),
            listen_address: default_listen_address(),
            enable_quic: true,
        }
    }
}

/// A list of manually-declared external multiaddr strings. Accepts EITHER a
/// single string (`external_address = "/dns4/.../tcp/8810"`) or a list
/// (`external_addresses = ["...", "..."]`) in TOML, so a one-address config
/// stays terse while multi-transport advertising (TCP + QUIC) is possible.
#[derive(Clone, Debug, Default, Serialize)]
pub struct ExternalAddresses(pub Vec<String>);

impl<'de> Deserialize<'de> for ExternalAddresses {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum OneOrMany {
            One(String),
            Many(Vec<String>),
        }
        Ok(match OneOrMany::deserialize(deserializer)? {
            OneOrMany::One(s) => ExternalAddresses(vec![s]),
            OneOrMany::Many(v) => ExternalAddresses(v),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The connection ceiling tightens as the contribution mode drops, and the
    /// default mode (Minimal) is well below the absolute ceiling.
    ///
    /// Before `max_peers` was wired up every node got a hardcoded 500
    /// regardless of contribution, so the Minimal assertion here is the one
    /// that fails if the resolution is removed.
    #[test]
    fn connection_ceiling_scales_with_contribution() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.max_peers, None, "default must be auto, not a number");

        let minimal = cfg.effective_max_connections(ContributionMode::Minimal);
        let moderate = cfg.effective_max_connections(ContributionMode::Moderate);
        let maximum = cfg.effective_max_connections(ContributionMode::Maximum);

        assert!(
            minimal < moderate && moderate < maximum,
            "contribution must tighten the ceiling monotonically: {minimal} / {moderate} / {maximum}"
        );
        assert!(
            minimal < MAX_ESTABLISHED_CONNECTIONS_CEILING,
            "the default mode must not hand out the absolute ceiling"
        );
        assert_eq!(maximum, MAX_ESTABLISHED_CONNECTIONS_CEILING);
        // Low enough to protect a home machine, high enough to still route:
        // gossipsub needs ~12 mesh peers and Kademlia ~20-40 contacts.
        assert!(
            minimal >= 100,
            "{minimal} is too low to hold a healthy DHT routing table"
        );
    }

    /// An explicit `max_peers` is the user's decision and wins in BOTH
    /// directions — the same rule `max_gpu_vram_mb` and `max_bandwidth_mbps`
    /// follow. Only a literal 0 is refused, since it isolates the node.
    #[test]
    fn explicit_max_peers_overrides_contribution() {
        let low = NetworkConfig {
            max_peers: Some(12),
            ..Default::default()
        };
        assert_eq!(low.effective_max_connections(ContributionMode::Maximum), 12);

        let high = NetworkConfig {
            max_peers: Some(900),
            ..Default::default()
        };
        assert_eq!(
            high.effective_max_connections(ContributionMode::Minimal),
            900,
            "an explicit ceiling above the default must not be clamped down"
        );

        let zero = NetworkConfig {
            max_peers: Some(0),
            ..Default::default()
        };
        assert_eq!(zero.effective_max_connections(ContributionMode::Minimal), 1);
    }

    #[test]
    fn wslinfo_mode_parsing() {
        // Mirrored — the only value that must skip the NAT-mode safe defaults.
        assert_eq!(parse_wslinfo_networking_mode("mirrored\n"), Some(true));
        assert_eq!(parse_wslinfo_networking_mode("  mirrored  "), Some(true));
        assert_eq!(parse_wslinfo_networking_mode("MIRRORED"), Some(true));
        // Any other explicit mode keeps the safe defaults.
        assert_eq!(parse_wslinfo_networking_mode("nat\n"), Some(false));
        assert_eq!(parse_wslinfo_networking_mode("none"), Some(false));
        assert_eq!(parse_wslinfo_networking_mode("virtioproxy"), Some(false));
        // No usable output → caller falls back to the loopback0 signal.
        assert_eq!(parse_wslinfo_networking_mode(""), None);
        assert_eq!(parse_wslinfo_networking_mode("   \n"), None);
    }

    #[test]
    fn upnp_and_external_address_defaults() {
        let cfg = NetworkConfig::default();
        // UPnP is on by default — the zero-config internet-reachability path.
        assert!(cfg.enable_upnp);
        // No external address is declared by default; discovery handles it.
        assert!(cfg.external_addresses.0.is_empty());
    }

    #[test]
    fn enable_upnp_defaults_true_when_absent_from_toml() {
        // A config file that predates the enable_upnp field must default to on.
        let cfg: NetworkConfig = toml::from_str("bootstrap_peers = []").unwrap();
        assert!(cfg.enable_upnp);
        assert!(cfg.external_addresses.0.is_empty());
    }

    #[test]
    fn bootstrap_peers_default_includes_anchor() {
        // Fresh installs (no config file) auto-join via the seed anchor.
        let cfg = NetworkConfig::default();
        assert!(!cfg.bootstrap_peers.is_empty());
        assert!(cfg
            .bootstrap_peers
            .iter()
            .any(|p| p.contains("swarmllm.duckdns.org")));
        // A config that omits the field also gets the default (serde default fn),
        // while an explicit empty list opts out.
        let omitted: NetworkConfig = toml::from_str("enable_relay = true").unwrap();
        assert!(!omitted.bootstrap_peers.is_empty());
        let explicit: NetworkConfig = toml::from_str("bootstrap_peers = []").unwrap();
        assert!(explicit.bootstrap_peers.is_empty());
    }

    #[test]
    fn external_addresses_accepts_single_string_or_list() {
        // Backward-compatible single string (via the `external_address` alias).
        let one: NetworkConfig =
            toml::from_str(r#"external_address = "/dns4/a.example/tcp/8810""#).unwrap();
        assert_eq!(one.external_addresses.0, vec!["/dns4/a.example/tcp/8810"]);

        // New list form — advertise the same host on TCP + QUIC.
        let many: NetworkConfig = toml::from_str(
            "external_addresses = [\"/dns4/a.example/tcp/8810\", \"/dns4/a.example/udp/8800/quic-v1\"]",
        )
        .unwrap();
        assert_eq!(many.external_addresses.0.len(), 2);
    }
}
