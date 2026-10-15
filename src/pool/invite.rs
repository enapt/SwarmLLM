//! Pool invite code v2 — `swarmpool://...` shareable bootstrap blob.
//!
//! The original 8-character invite code (still kept inside this blob as the
//! actual join token) only worked once both nodes were already on the same
//! libp2p swarm — it carried no information about how to *find* the inviter.
//! In a fully decentralized network that's fine (DHT bootstrap), but during
//! the bootstrap-before-decentralization phase the pool owner is often the
//! only address the joiner can reach. v2 fixes that by bundling the owner's
//! reachable listen multiaddrs alongside the join token, so a fresh node
//! anywhere with the code can:
//!
//! 1. Decode it.
//! 2. Dial one of the owner's multiaddrs over the wire (Tailscale, LAN,
//!    public IP, whatever's in the bundle).
//! 3. Once a libp2p connection lands, broadcast the existing
//!    `PoolMessage::JoinRequest { code_hash, ... }` over GossipSub.
//!
//! The owner's handler still matches `code_hash` against its
//! `invite_codes` map exactly as before — there is **no wire-protocol
//! change** for the pool-join itself. v2 is purely the rendezvous wrapper.
//!
//! ## Wire format
//!
//! - prefix: `swarmpool://`
//! - payload: base64url(no-pad) of `key (32B) || nonce (12B) || ciphertext`
//! - ciphertext: ChaCha20-Poly1305 of `serde_json::to_vec(&InviteCodePayload)`
//!
//! Encryption is here to prevent casual IP harvesting from anyone glancing
//! at a code pasted into a chat window. The key is embedded in the code, so
//! anyone with the full code can decrypt — same threat model as the existing
//! `swarm://` network code in `src/network/discovery.rs`.

use base64::Engine;
use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::types::NodeId;

/// `swarmpool://...` is the v2 prefix. Separate from `swarm://` (network-only
/// invite) so callers can tell at a glance which flow a pasted code drives.
pub const INVITE_PREFIX: &str = "swarmpool://";

/// Current wire-format version. Bump when changing the inner payload.
pub const INVITE_VERSION: u8 = 1;

/// Hard cap on the encoded length we'll attempt to decode. A typical blob is
/// ~300-500 chars; anything bigger is either garbage or an attempted DoS.
const MAX_ENCODED_LEN: usize = 2048;

/// Inner payload of a v2 invite code — what the recipient gets after
/// decryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InviteCodePayload {
    /// Wire-format version. Must equal `INVITE_VERSION` or decode fails fast.
    pub version: u8,
    /// The pool ID. Equals the owner's `NodeId`; the libp2p `PeerId` is
    /// derivable from this via `peer_id_to_node_id`'s inverse.
    pub pool_id: NodeId,
    /// Optional human-friendly pool name for the joining UI. Empty string if
    /// the owner hasn't named the pool.
    pub pool_name: String,
    /// The owner's reachable listen multiaddrs (each terminated with
    /// `/p2p/<peer_id>` so a remote dialer can verify the target identity).
    /// The joiner dials each in order until one connects.
    pub multiaddrs: Vec<String>,
    /// The 8-char short code that still drives the actual join wire protocol.
    /// The joiner hashes this and broadcasts `JoinRequest { code_hash, ... }`
    /// over GossipSub — same as the legacy 8-char path.
    pub code: String,
    /// Expiry as a unix-seconds timestamp. Mirrored from the underlying
    /// `PoolInviteCode.expires_at` so decoders can fail fast before dialing.
    pub expires_at_unix: i64,
}

impl InviteCodePayload {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Encode an invite code into the `swarmpool://...` wire form.
///
/// Returns `Err` only if JSON serialization of the payload fails — in
/// practice that means a bug, since all field types are infallibly
/// serializable.
pub fn encode_invite_code(payload: &InviteCodePayload) -> Result<String, SwarmError> {
    let plaintext = serde_json::to_vec(payload)
        .map_err(|e| SwarmError::Internal(format!("invite payload serialize: {e}")))?;

    let mut key_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut key_bytes);
    OsRng.fill_bytes(&mut nonce_bytes);

    let cipher = ChaCha20Poly1305::new_from_slice(&key_bytes).expect("valid key size");
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_slice())
        .expect("ChaCha20Poly1305 encryption cannot fail on a valid key");

    let mut packed = Vec::with_capacity(32 + 12 + ciphertext.len());
    packed.extend_from_slice(&key_bytes);
    packed.extend_from_slice(&nonce_bytes);
    packed.extend_from_slice(&ciphertext);
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&packed);
    Ok(format!("{INVITE_PREFIX}{encoded}"))
}

/// Decode a `swarmpool://...` blob back to its inner payload.
///
/// Returns `SwarmError::Validation` for malformed input, expired codes, or
/// version mismatches — anything the user might paste that we can recognize
/// and report cleanly. Decryption failure also surfaces as `Validation`
/// (rather than `Internal`) because the most likely cause is a truncated
/// or mistyped code, not a daemon bug.
pub fn decode_invite_code(raw: &str) -> Result<InviteCodePayload, SwarmError> {
    let trimmed = raw.trim();
    let Some(encoded) = trimmed.strip_prefix(INVITE_PREFIX) else {
        return Err(SwarmError::Validation(format!(
            "Invite code must start with '{INVITE_PREFIX}'"
        )));
    };

    if encoded.is_empty() {
        return Err(SwarmError::Validation("Invite code is empty".into()));
    }
    if encoded.len() > MAX_ENCODED_LEN {
        return Err(SwarmError::Validation(format!(
            "Invite code too long (max {MAX_ENCODED_LEN} chars)"
        )));
    }

    // Every way the bytes can be wrong means one thing to the person holding
    // the code — it did not survive being copied — and they can act on exactly
    // one remedy. Passing the library's own text through instead produced
    // "Invite code is not valid base64url: Invalid last symbol 101, offset 6"
    // beside a sibling arm that reads "is corrupt or was edited — ask the
    // inviter to regenerate", so which of the two a user got depended on HOW
    // their paste was mangled. The detail is logged, not shown.
    let packed = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .map_err(|e| {
            tracing::debug!(error = %e, "invite code failed base64url decode");
            SwarmError::Validation(CORRUPT_CODE_MESSAGE.into())
        })?;

    if packed.len() < 32 + 12 + 16 {
        return Err(SwarmError::Validation(
            "Invite code is too short to be valid".into(),
        ));
    }

    let key_bytes = &packed[..32];
    let nonce_bytes = &packed[32..44];
    let ciphertext = &packed[44..];

    let cipher = ChaCha20Poly1305::new_from_slice(key_bytes).map_err(|e| {
        tracing::debug!(error = %e, "invite code embedded key rejected");
        SwarmError::Validation(CORRUPT_CODE_MESSAGE.into())
    })?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| SwarmError::Validation(CORRUPT_CODE_MESSAGE.into()))?;

    let payload: InviteCodePayload = serde_json::from_slice(&plaintext).map_err(|e| {
        tracing::debug!(error = %e, "invite code payload did not parse after decryption");
        SwarmError::Validation(CORRUPT_CODE_MESSAGE.into())
    })?;

    if payload.version != INVITE_VERSION {
        return Err(SwarmError::Validation(format!(
            "Unsupported invite code version {} (this daemon supports v{})",
            payload.version, INVITE_VERSION
        )));
    }
    if payload.is_expired() {
        return Err(SwarmError::Validation(
            "Invite code has expired — ask the inviter to generate a new one".into(),
        ));
    }
    if payload.code.len() != 8 {
        return Err(SwarmError::Validation(
            "Invite code payload is malformed (join token wrong length)".into(),
        ));
    }
    Ok(payload)
}

/// What every "these bytes are not a usable code" failure tells the user.
///
/// Shared so the message cannot drift between the ways a paste can be mangled:
/// truncated, re-wrapped by a chat client, edited by hand, or corrupted in
/// transit all leave the reader with the same single remedy. The specific cause
/// goes to `debug!` — useful to whoever is debugging, meaningless to whoever is
/// joining a pool.
pub(crate) const CORRUPT_CODE_MESSAGE: &str =
    "Invite code is corrupt or was edited — ask the inviter to regenerate";

/// Quick sniff: does this string look like a v2 code? Used by the API layer
/// to route a pasted blob to either the v2 path or the legacy 8-char path.
pub fn looks_like_v2(raw: &str) -> bool {
    raw.trim().starts_with(INVITE_PREFIX)
}

/// Does at least one of these multiaddrs denote an address reachable from the
/// **open internet** (as opposed to LAN-only)?
///
/// Distinct from `network::manager::events::addr_is_remotely_reachable`, which
/// keeps LAN + CGNAT/Tailscale addresses (reachable on the same overlay). This
/// is the stricter test used to decide whether an invite code will actually
/// work for a stranger somewhere else on the internet — if it returns false,
/// the code is LAN-only and we warn the user rather than let it fail silently.
///
/// True for: global/public IPv4 & IPv6, any DNS name (`/dns4`, `/dns6`,
/// `/dnsaddr` — a dynamic-DNS anchor), and relay-circuit addresses
/// (`/p2p-circuit`). False for RFC1918 LAN, loopback, link-local, IPv6 ULA,
/// and CGNAT/Tailscale (100.64.0.0/10) — those only work locally or within a
/// private overlay.
pub fn any_internet_reachable(multiaddrs: &[String]) -> bool {
    multiaddrs
        .iter()
        .any(|s| match s.parse::<libp2p::Multiaddr>() {
            Ok(m) => multiaddr_is_internet_reachable(&m),
            Err(_) => false,
        })
}

/// The addresses an invite code carries: the ones a joiner can use.
///
/// That is this computer's address on the network its traffic leaves by
/// (`primary`, [`outbound_addresses`]) — the LAN a device beside it shares —
/// and every address reachable from the internet (public, DNS name, relay
/// circuit). Every other local address goes: a Docker bridge (`172.17.0.1`,
/// reachable only from containers on this machine, and on a joiner's machine
/// its OWN bridge), a VPN overlay such as Tailscale (`100.64.0.0/10`) unless
/// it carries this computer's traffic, a second network card. Kept, they told
/// whoever got the code about this computer's VPN and containers, and made a
/// 1,592-character code of ten addresses where four did anything (field report
/// 2026-10-15). A device on the overlay still reaches the owner by the
/// internet or relay route, and learns every address once connected.
///
/// With no `primary` known (no route out: an offline LAN), or nothing left
/// after the cut, every address stays — a code that works where it did.
///
/// When the traffic leaves by an overlay — a Tailscale exit node, whose
/// address is CGNAT (`100.64.0.0/10`), with no private-LAN address on the
/// route — the LAN a device beside this one shares is not the route out, so
/// the private-LAN addresses stay too, less Docker's default bridge
/// (`172.17.0.0/16`). A full-tunnel VPN that hands out a private address
/// (WireGuard's `10.x`) cannot be told from a LAN this way; a device beside
/// it joins by the internet or relay route (pre-release security review,
/// 2026-10-15).
pub fn addresses_for_invite(all: &[String], primary: &[std::net::IpAddr]) -> Vec<String> {
    use libp2p::multiaddr::Protocol;
    use std::net::IpAddr;
    if primary.is_empty() {
        return all.to_vec();
    }
    let leaves_by_an_overlay = primary
        .iter()
        .any(|ip| matches!(ip, IpAddr::V4(v4) if is_cgnat_ipv4(*v4)))
        && !primary
            .iter()
            .any(|ip| matches!(ip, IpAddr::V4(v4) if v4.is_private()));
    let kept: Vec<String> = all
        .iter()
        .filter(|s| {
            let Ok(m) = s.parse::<libp2p::Multiaddr>() else {
                return false;
            };
            multiaddr_is_internet_reachable(&m)
                || m.iter().any(|p| match p {
                    Protocol::Ip4(ip) => {
                        primary.contains(&IpAddr::V4(ip))
                            || (leaves_by_an_overlay
                                && ip.is_private()
                                && !is_docker_default_bridge(ip))
                    }
                    Protocol::Ip6(ip) => primary.contains(&IpAddr::V6(ip)),
                    _ => false,
                })
        })
        .cloned()
        .collect();
    if kept.is_empty() {
        all.to_vec()
    } else {
        kept
    }
}

/// This computer's own address on the route its traffic to the internet
/// takes, per address family — the interface holding the default route.
///
/// Asked of the routing table, not the network: `connect` on a UDP socket
/// sends nothing, it only fixes the socket's source address the way the
/// kernel would route a datagram (the same question `ip route get 1.1.1.1`
/// answers). Docker bridges and VPN overlays that do not carry the default
/// route are never the answer. Empty with no route out.
pub fn outbound_addresses() -> Vec<std::net::IpAddr> {
    let probe = |bind: &str, target: &str| -> Option<std::net::IpAddr> {
        let socket = std::net::UdpSocket::bind(bind).ok()?;
        socket.connect(target).ok()?;
        let ip = socket.local_addr().ok()?.ip();
        (!ip.is_unspecified() && !ip.is_loopback()).then_some(ip)
    };
    [
        probe("0.0.0.0:0", "1.1.1.1:53"),
        probe("[::]:0", "[2606:4700:4700::1111]:53"),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// Whether a multiaddr is reachable from the open internet.
///
/// Public IP or DNS name, or a relay circuit. Deliberately EXCLUDES private
/// ranges, link-local, loopback and CGNAT (100.64.0.0/10, which Tailscale hands
/// out) — those are reachable from somewhere, but not from the internet.
///
/// Shared with the AutoNAT handler, which must not accept a LAN peer's
/// successful probe of a LAN address as proof of internet reachability. See
/// `network::manager::events`.
pub(crate) fn multiaddr_is_internet_reachable(addr: &libp2p::Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut via_relay = false;
    for proto in addr.iter() {
        match proto {
            Protocol::P2pCircuit => via_relay = true,
            Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) | Protocol::Dnsaddr(_) => {
                return true
            }
            Protocol::Ip4(ip) if is_public_ipv4(ip) => return true,
            Protocol::Ip6(ip) if is_public_ipv6(ip) => return true,
            _ => {}
        }
    }
    via_relay
}

fn is_public_ipv4(ip: std::net::Ipv4Addr) -> bool {
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || is_cgnat_ipv4(ip))
}

/// RFC 6598 shared address space (100.64.0.0/10) — carrier-grade NAT and the
/// range Tailscale hands out. Reachable within the overlay, not the internet.
fn is_cgnat_ipv4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (o[1] & 0xc0) == 0x40
}

/// Docker's default bridge (`docker0`, 172.17.0.0/16): reachable only from
/// this machine's containers, and on a joiner's machine its OWN bridge.
fn is_docker_default_bridge(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 172 && o[1] == 17
}

fn is_public_ipv6(ip: std::net::Ipv6Addr) -> bool {
    let seg0 = ip.segments()[0];
    !(ip.is_loopback()
        || ip.is_unspecified()
        || (seg0 & 0xffc0) == 0xfe80  // link-local fe80::/10
        || (seg0 & 0xfe00) == 0xfc00) // unique-local fc00::/7
}

#[cfg(test)]
mod tests {

    /// However a pasted code is mangled, the person holding it gets one message
    /// and one remedy.
    ///
    /// The arms used to disagree: a base64 fault surfaced the library's own text
    /// ("Invalid last symbol 101, offset 6") while a decryption fault said
    /// "corrupt or was edited — ask the inviter to regenerate". Which one a user
    /// saw depended on HOW their paste broke, and only one of the two could be
    /// acted on.
    #[test]
    fn every_mangled_code_gives_the_same_actionable_message() {
        let cases = [
            ("swarmpool://garbage", "non-base64 characters"),
            ("swarmpool://QUJDREVGRw", "decodes but far too short"),
            (
                "swarmpool://AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "right shape, will not decrypt",
            ),
        ];
        for (code, why) in cases {
            let err = decode_invite_code(code).expect_err(why);
            let msg = err.to_string();
            assert!(
                matches!(err, SwarmError::Validation(_)),
                "{why}: a bad paste is the caller's input, not a daemon fault — got {err:?}"
            );
            assert!(
                !msg.contains("offset") && !msg.contains("symbol") && !msg.contains("base64"),
                "{why}: leaks library detail a user cannot act on: {msg}"
            );
        }
    }
    use super::*;

    /// **A LAN peer confirming a LAN address does not make this node public.**
    ///
    /// AutoNAT servers are ordinary peers, so a node on the same subnet happily
    /// confirms an RFC1918 address. Treating that as "Public" makes a NAT'd node
    /// skip reserving a relay and sit unreachable from the internet while
    /// reporting otherwise. Observed live 2026-08-05: a node reporting
    /// `nat: Public` on the strength of confirmations for these exact addresses.
    #[test]
    fn a_confirmed_private_address_is_not_internet_reachable() {
        for addr in [
            "/ip4/192.168.1.53/tcp/8810",   // RFC1918, confirmed by a LAN peer
            "/ip4/10.255.255.254/tcp/8810", // RFC1918
            "/ip4/169.254.83.107/tcp/8810", // link-local — never routable
            "/ip4/172.17.0.1/tcp/8810",     // Docker bridge
            "/ip4/127.0.0.1/tcp/8810",      // loopback
            "/ip4/100.64.0.7/tcp/8810",     // CGNAT / Tailscale
        ] {
            let m: libp2p::Multiaddr = addr.parse().unwrap();
            assert!(
                !multiaddr_is_internet_reachable(&m),
                "{addr} must not count as internet-reachable"
            );
        }
    }

    /// The addresses that genuinely do mean public reachability.
    #[test]
    fn public_addresses_and_relay_circuits_are_internet_reachable() {
        for addr in [
            "/ip4/203.0.113.7/tcp/8810",
            "/dns4/anchor.example.net/tcp/8810",
        ] {
            let m: libp2p::Multiaddr = addr.parse().unwrap();
            assert!(
                multiaddr_is_internet_reachable(&m),
                "{addr} should count as internet-reachable"
            );
        }
    }

    fn sample_payload() -> InviteCodePayload {
        InviteCodePayload {
            version: INVITE_VERSION,
            pool_id: NodeId([7u8; 32]),
            pool_name: "Test Pool".into(),
            multiaddrs: vec![
                "/ip4/100.64.0.5/tcp/8810/p2p/12D3KooWFakeXXX".into(),
                "/ip4/192.168.1.5/udp/8800/quic-v1/p2p/12D3KooWFakeXXX".into(),
            ],
            code: "A3F7K2M9".into(),
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
        }
    }

    #[test]
    fn roundtrip_preserves_payload() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        assert!(encoded.starts_with(INVITE_PREFIX));

        let decoded = decode_invite_code(&encoded).unwrap();
        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.pool_id, original.pool_id);
        assert_eq!(decoded.pool_name, original.pool_name);
        assert_eq!(decoded.multiaddrs, original.multiaddrs);
        assert_eq!(decoded.code, original.code);
        assert_eq!(decoded.expires_at_unix, original.expires_at_unix);
    }

    #[test]
    fn whitespace_around_code_is_tolerated() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        let padded = format!("  \n{encoded}\t  ");
        let decoded = decode_invite_code(&padded).unwrap();
        assert_eq!(decoded.code, original.code);
    }

    #[test]
    fn missing_prefix_fails() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let stripped = encoded.strip_prefix(INVITE_PREFIX).unwrap();
        let err = decode_invite_code(stripped).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn truncated_code_fails_cleanly() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let truncated = &encoded[..encoded.len() - 5];
        let err = decode_invite_code(truncated).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn tampered_ciphertext_fails_authentication() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        // Flip the last char of the base64url body to a different valid char.
        let mut bytes: Vec<char> = encoded.chars().collect();
        let last = bytes.last_mut().unwrap();
        *last = if *last == 'A' { 'B' } else { 'A' };
        let tampered: String = bytes.into_iter().collect();
        let err = decode_invite_code(&tampered).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn expired_code_rejected() {
        let mut payload = sample_payload();
        payload.expires_at_unix = chrono::Utc::now().timestamp() - 60;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("expired"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn version_mismatch_rejected() {
        let mut payload = sample_payload();
        payload.version = INVITE_VERSION + 7;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("version"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn malformed_join_token_length_rejected() {
        let mut payload = sample_payload();
        payload.code = "TOOSHORT".into(); // 8 — should pass
        let encoded = encode_invite_code(&payload).unwrap();
        decode_invite_code(&encoded).unwrap();

        payload.code = "WAYTOOLONG".into(); // 10
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn oversized_input_rejected_before_decode() {
        let huge = format!("{INVITE_PREFIX}{}", "A".repeat(MAX_ENCODED_LEN + 1));
        let err = decode_invite_code(&huge).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("too long"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn looks_like_v2_sniff() {
        assert!(looks_like_v2("swarmpool://abc"));
        assert!(looks_like_v2("  swarmpool://xyz  "));
        assert!(!looks_like_v2("A3F7K2M9"));
        assert!(!looks_like_v2("swarm://abc"));
        assert!(!looks_like_v2(""));
    }

    // A real, parseable peer id — the fixtures below carry a valid `/p2p/...`
    // suffix exactly like production `listen_multiaddrs` entries do, so the
    // classifier is exercised on strings that actually parse.
    fn pid() -> libp2p::PeerId {
        libp2p::PeerId::random()
    }

    #[test]
    fn internet_reachable_public_ipv4() {
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/ip4/203.0.113.5/tcp/8810/p2p/{p}"
        )]));
        assert!(any_internet_reachable(&[format!(
            "/ip4/8.8.8.8/udp/8800/quic-v1/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_dns_anchor() {
        // A dynamic-DNS anchor address is internet-reachable regardless of the
        // IP it currently resolves to.
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/dns4/anchor.example.net/tcp/8810/p2p/{p}"
        )]));
        assert!(any_internet_reachable(&[format!(
            "/dns6/anchor.example.net/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_relay_circuit() {
        let (relay, me) = (pid(), pid());
        assert!(any_internet_reachable(&[format!(
            "/ip4/203.0.113.9/tcp/8810/p2p/{relay}/p2p-circuit/p2p/{me}"
        )]));
    }

    #[test]
    fn internet_reachable_public_ipv6() {
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/ip6/2001:db8::1/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn not_internet_reachable_lan_only() {
        // RFC1918 LAN, loopback, link-local, IPv6 ULA — none reachable from
        // the open internet.
        let p = pid();
        assert!(!any_internet_reachable(&[
            format!("/ip4/192.168.1.5/tcp/8810/p2p/{p}"),
            format!("/ip4/10.0.0.7/udp/8800/quic-v1/p2p/{p}"),
            format!("/ip4/127.0.0.1/tcp/8810/p2p/{p}"),
            format!("/ip6/fc00::1/tcp/8810/p2p/{p}"),
        ]));
    }

    #[test]
    fn not_internet_reachable_cgnat_tailscale() {
        // 100.64.0.0/10 — CGNAT / Tailscale. Works on the overlay, not the
        // open internet, so an invite with only these must warn.
        let p = pid();
        assert!(!any_internet_reachable(&[format!(
            "/ip4/100.64.10.5/tcp/8810/p2p/{p}"
        )]));
        assert!(!any_internet_reachable(&[format!(
            "/ip4/100.127.255.254/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_mixed_list_true_if_any() {
        // A LAN address alongside a public one still counts as reachable.
        let p = pid();
        assert!(any_internet_reachable(&[
            format!("/ip4/192.168.1.5/tcp/8810/p2p/{p}"),
            format!("/dns4/anchor.example.net/tcp/8810/p2p/{p}"),
        ]));
    }

    #[test]
    fn not_internet_reachable_empty_or_garbage() {
        assert!(!any_internet_reachable(&[]));
        assert!(!any_internet_reachable(&["not a multiaddr".into()]));
    }

    /// **An invite code carries the addresses a joiner can use, and no
    /// others** (field report 2026-10-15): the owner's code listed its
    /// Tailscale address and two Docker bridges beside its LAN and public
    /// ones, on both transports — ten addresses where four did anything.
    #[test]
    fn an_invite_code_carries_only_addresses_a_joiner_can_use() {
        let id = "/p2p/12D3KooWQHiVu2JpLSVrWUCRmbcz1fbNc4tg1kxBRkGNw8kXXeNj";
        let all: Vec<String> = [
            "100.101.102.103",
            "172.17.0.1",
            "172.18.0.1",
            "192.168.1.20",
            "81.2.69.160",
        ]
        .iter()
        .flat_map(|ip| {
            [
                format!("/ip4/{ip}/tcp/8810{id}"),
                format!("/ip4/{ip}/udp/8800/quic-v1{id}"),
            ]
        })
        .collect();
        let lan: std::net::IpAddr = "192.168.1.20".parse().unwrap();
        let kept = super::addresses_for_invite(&all, &[lan]);
        assert_eq!(kept.len(), 4, "{kept:#?}");
        assert!(kept
            .iter()
            .all(|a| a.contains("/192.168.1.20/") || a.contains("/81.2.69.160/")));

        // A relay circuit is a way in from the internet.
        let relayed = format!("/ip4/81.2.69.160/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G/p2p-circuit{id}");
        assert_eq!(
            super::addresses_for_invite(std::slice::from_ref(&relayed), &[lan]),
            vec![relayed]
        );

        // Traffic out by a Tailscale exit node: the LAN a device beside this
        // one shares is not the route out, so it stays — Docker's default
        // bridge still does not.
        let exit_node: std::net::IpAddr = "100.101.102.103".parse().unwrap();
        let kept = super::addresses_for_invite(&all, &[exit_node]);
        assert!(
            kept.iter().any(|a| a.contains("/192.168.1.20/")),
            "{kept:#?}"
        );
        assert!(
            !kept.iter().any(|a| a.contains("/172.17.0.1/")),
            "{kept:#?}"
        );
        assert!(kept.iter().any(|a| a.contains("/100.101.102.103/")));
        // A LAN on the route out keeps the overlay's private addresses out.
        let kept = super::addresses_for_invite(&all, &[lan, exit_node]);
        assert!(
            !kept.iter().any(|a| a.contains("/172.18.0.1/")),
            "{kept:#?}"
        );

        // No route out to tell the LAN by, or nothing left: every address stays.
        assert_eq!(super::addresses_for_invite(&all, &[]), all);
        let private_only = all[..6].to_vec();
        assert_eq!(
            super::addresses_for_invite(&private_only, &[lan]),
            private_only
        );
    }
}
