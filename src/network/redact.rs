//! Making a diagnostics report safe to paste in public.
//!
//! `GET /api/admin/diagnostics` exists in order to be *shared*. The dashboard
//! has a one-click "Copy diagnostics" button, the README calls the report "the
//! single most useful thing to include in a bug report", and the person
//! clicking that button is explicitly not expected to read what it copied —
//! the whole point is that they do not have to.
//!
//! What it copied included every address the node knows: its own, and up to
//! ten remembered *dialable* peer multiaddrs, which on a live node are other
//! people's home IP addresses. The button's hint read "No keys or invite codes
//! are included", which a non-technical reader reasonably takes to mean the
//! text is safe to paste into a public channel.
//!
//! So the report is redacted before it leaves the daemon, and `?full=1` is a
//! deliberate opt-in for an operator debugging their own machine.
//!
//! **The redaction is one pass over the finished report, not a rule applied
//! per section.** Addresses reach the text from at least four places — the
//! listen-address list, the peer cache, relay-circuit hops, and the prose of
//! whatever error a failed dial produced — and a per-section rule is the shape
//! this codebase keeps getting caught by (see `.claude/rules/architecture.md`
//! § "One invariant, N paths"). A section added later inherits the redaction
//! with no author action.
//!
//! **A redacted address keeps everything a reading depends on.** Transport,
//! port, peer id and the `/p2p-circuit` structure that says a hop is relayed
//! all survive; only the host is replaced, by a placeholder naming its KIND —
//! so "this node has no public address" and "the relay hop is the project
//! anchor" are both still legible. Deleting the lines would have destroyed
//! that, and a report nobody can read is not safer, it just gets pasted with
//! `?full=1` instead.
//!
//! **Peer ids and node ids are deliberately NOT redacted.** They are the
//! swarm's public identities, they appear unredacted throughout the rest of
//! the report, and they are not a coordinate anyone can dial. The address is.

use std::collections::HashSet;
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::OnceLock;

/// Punctuation that can sit in front of an address in prose without being part
/// of it. `[` is deliberately absent — it introduces the bracketed IPv6 form,
/// which [`redact_bare_host`] handles itself.
const LEADING_PUNCT: &[char] = &['(', '"', '\''];
/// Punctuation that can trail an address in prose without being part of it.
const TRAILING_PUNCT: &[char] = &[',', '.', ';', ':', ')', ']', '"', '\'', '!', '?'];

/// Multiaddr protocol names whose following component is a host we must hide.
const HOST_PROTOCOLS: &[&str] = &["ip4", "ip6", "dns", "dns4", "dns6", "dnsaddr"];

/// Replace every host that identifies a *machine* with a placeholder naming
/// its kind, leaving the rest of the report untouched.
///
/// The tag appended to each placeholder is stable within one report and
/// meaningless outside it, so a reader can still tell "ten cache entries, all
/// the same host" from "ten different hosts" — which is most of what the peer
/// cache section is for.
pub fn redact_addresses(text: &str) -> String {
    use rand::RngCore;
    let mut salt = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut salt);
    redact_with_salt(text, &salt)
}

/// The salt is a parameter so tests can assert exact output. It is random per
/// report in production **because IPv4 is only 2^32 addresses** — an unsalted
/// digest of one is recovered by enumeration in seconds, which would make the
/// placeholder a reversible encoding of the thing it is hiding rather than a
/// redaction of it.
fn redact_with_salt(text: &str, salt: &[u8; 16]) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    // `split_inclusive` keeps the whitespace attached, so reassembling the
    // pieces reproduces the input byte for byte wherever nothing matched.
    for piece in text.split_inclusive(char::is_whitespace) {
        let body = piece.trim_end();
        let space = &piece[body.len()..];
        let unpunctuated = body.trim_end_matches(TRAILING_PUNCT);
        let trailing = &body[unpunctuated.len()..];
        let core = unpunctuated.trim_start_matches(LEADING_PUNCT);
        let leading = &unpunctuated[..unpunctuated.len() - core.len()];

        out.push_str(leading);
        match redact_token(core, salt) {
            Some(replaced) => out.push_str(&replaced),
            None => out.push_str(core),
        }
        out.push_str(trailing);
        out.push_str(space);
    }
    out
}

/// `None` means "nothing in this token identifies a machine" — the caller
/// then emits it verbatim.
fn redact_token(core: &str, salt: &[u8; 16]) -> Option<String> {
    if core.is_empty() {
        return None;
    }
    if core.contains('/') {
        return redact_multiaddr(core, salt);
    }
    redact_bare_host(core, salt)
}

/// Walk a multiaddr's `/`-separated components and replace only the ones a
/// host protocol introduces. Anything else — transports, ports, `p2p` peer
/// ids, `p2p-circuit` — is copied through, which is what keeps a redacted
/// relay circuit readable as a relay circuit.
fn redact_multiaddr(core: &str, salt: &[u8; 16]) -> Option<String> {
    let parts: Vec<&str> = core.split('/').collect();
    let mut changed = false;
    let mut out: Vec<String> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let introduced_by = if i == 0 { "" } else { parts[i - 1] };
        let replacement = if !HOST_PROTOCOLS.contains(&introduced_by) {
            None
        } else if introduced_by.starts_with("dns") {
            dns_placeholder(part, salt)
        } else {
            ip_placeholder(part, salt)
        };
        match replacement {
            Some(r) => {
                changed = true;
                out.push(r);
            }
            None => out.push((*part).to_string()),
        }
    }
    changed.then(|| out.join("/"))
}

/// Bare hosts as they appear in prose — a dial error, a log line quoted into
/// the report. `1.2.3.4`, `1.2.3.4:8810`, `[fe80::1]:8810`, `fe80::1`.
fn redact_bare_host(core: &str, salt: &[u8; 16]) -> Option<String> {
    if let Some(rest) = core.strip_prefix('[') {
        if let Some((host, tail)) = rest.split_once(']') {
            let placeholder = ip_placeholder(host, salt)?;
            return Some(format!("[{placeholder}]{tail}"));
        }
    }
    if let Some((host, port)) = core.rsplit_once(':') {
        let numeric_port = !port.is_empty() && port.chars().all(|c| c.is_ascii_digit());
        if numeric_port && host.parse::<Ipv4Addr>().is_ok() {
            let placeholder = ip_placeholder(host, salt)?;
            return Some(format!("{placeholder}:{port}"));
        }
    }
    ip_placeholder(core, salt)
}

/// `None` for anything that is not an IP, and for the ones that identify
/// nobody: loopback and the unspecified address carry no information about a
/// machine, and leaving them literal keeps the report readable.
fn ip_placeholder(value: &str, salt: &[u8; 16]) -> Option<String> {
    if is_published_anchor_host(value) {
        return None;
    }
    let kind = if let Ok(v4) = value.parse::<Ipv4Addr>() {
        if v4.is_loopback() || v4.is_unspecified() {
            return None;
        }
        ipv4_kind(v4)
    } else if let Ok(v6) = value.parse::<Ipv6Addr>() {
        if v6.is_loopback() || v6.is_unspecified() {
            return None;
        }
        ipv6_kind(v6)
    } else {
        return None;
    };
    Some(format!("<{kind}-{}>", tag(value, salt)))
}

fn dns_placeholder(value: &str, salt: &[u8; 16]) -> Option<String> {
    if value.is_empty() || is_published_anchor_host(value) {
        return None;
    }
    Some(format!("<host-{}>", tag(value, salt)))
}

/// The public/private boundary is taken from [`is_non_public_ipv4_bytes`],
/// which is already the single answer to that question for PEX filtering and
/// the anti-gaming subnet tracker. Two notions of "public" that disagreed
/// would be a way to leak an address the rest of the code calls public.
///
/// [`is_non_public_ipv4_bytes`]: crate::network::helpers::is_non_public_ipv4_bytes
fn ipv4_kind(ip: Ipv4Addr) -> &'static str {
    let octets = ip.octets();
    if octets[0] == 100 && (64..128).contains(&octets[1]) {
        // RFC 6598 shared address space — carrier-grade NAT, and what
        // Tailscale hands out. Worth naming separately: it explains a node
        // that is neither on your LAN nor reachable from the internet.
        "cgnat-ip"
    } else if crate::network::helpers::is_non_public_ipv4_bytes(&octets) {
        "private-ip"
    } else {
        "public-ip"
    }
}

fn ipv6_kind(ip: Ipv6Addr) -> &'static str {
    let first = ip.segments()[0];
    if (first & 0xffc0) == 0xfe80 || (first & 0xfe00) == 0xfc00 {
        "private-ip"
    } else {
        "public-ip"
    }
}

/// Hosts belonging to the project's own bootstrap anchor, which every binary
/// ships in plain text and the repository publishes. Redacting them protects
/// nobody and costs the most useful reading in the report — whether a node is
/// reaching the swarm through the anchor, and whether a relay hop is ours.
///
/// Derived from [`default_bootstrap_peers`] rather than restated, so moving
/// the anchor moves the exemption with it.
///
/// [`default_bootstrap_peers`]: crate::config::default_bootstrap_peers
fn is_published_anchor_host(value: &str) -> bool {
    static HOSTS: OnceLock<HashSet<String>> = OnceLock::new();
    HOSTS
        .get_or_init(|| {
            let mut set = HashSet::new();
            for addr in crate::config::default_bootstrap_peers() {
                let parts: Vec<&str> = addr.split('/').collect();
                for (i, part) in parts.iter().enumerate() {
                    if i > 0 && HOST_PROTOCOLS.contains(&parts[i - 1]) {
                        set.insert((*part).to_string());
                    }
                }
            }
            set
        })
        .contains(value)
}

/// Four hex characters of a salted digest — enough to tell two hosts apart
/// inside one report, and nothing at all outside it.
fn tag(value: &str, salt: &[u8; 16]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(salt);
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    format!("{:02x}{:02x}", bytes[0], bytes[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SALT: [u8; 16] = [7u8; 16];

    fn r(text: &str) -> String {
        redact_with_salt(text, &SALT)
    }

    #[test]
    fn a_peer_address_loses_its_host_and_keeps_everything_else() {
        let addr =
            "/ip4/81.241.51.1/tcp/8810/p2p/12D3KooWJAfgXRJSBLeRsUEvjVxRjsf6iaq6SNrjEVGAdkpnkF7J";
        let out = r(addr);
        // The control: without this pass the address is right there in the text
        // someone is about to paste into a public channel.
        assert!(addr.contains("81.241.51.1"));
        assert!(!out.contains("81.241.51.1"), "host survived: {out}");
        // Everything a reading of the report depends on is still there.
        assert!(out.contains("/tcp/8810"), "{out}");
        assert!(
            out.contains("12D3KooWJAfgXRJSBLeRsUEvjVxRjsf6iaq6SNrjEVGAdkpnkF7J"),
            "{out}"
        );
        assert!(out.contains("<public-ip-"), "kind not named: {out}");
    }

    #[test]
    fn the_placeholder_names_which_kind_of_address_it_replaced() {
        assert!(r("/ip4/192.168.1.57/tcp/8810").contains("<private-ip-"));
        assert!(r("/ip4/100.90.1.2/tcp/8810").contains("<cgnat-ip-"));
        assert!(r("/ip4/8.8.8.8/tcp/8810").contains("<public-ip-"));
        assert!(r("/ip6/2001:db8::1/tcp/8810").contains("<public-ip-"));
        assert!(r("/ip6/fe80::1/tcp/8810").contains("<private-ip-"));
        assert!(r("/dns4/example.org/tcp/8810").contains("<host-"));
    }

    #[test]
    fn an_address_that_identifies_nobody_is_left_alone() {
        // Loopback and the unspecified address say nothing about a machine,
        // and a report that hides them is harder to read for no gain.
        assert_eq!(r("/ip4/127.0.0.1/tcp/8810"), "/ip4/127.0.0.1/tcp/8810");
        assert_eq!(r("/ip4/0.0.0.0/tcp/8810"), "/ip4/0.0.0.0/tcp/8810");
        assert_eq!(r("/ip6/::1/tcp/8810"), "/ip6/::1/tcp/8810");
    }

    #[test]
    fn the_projects_own_anchor_is_not_redacted() {
        // It ships in every binary and is published in the repository, so
        // hiding it protects nobody — and "am I reaching the swarm through the
        // anchor?" is one of the questions the report exists to answer.
        let dns = "/dns4/swarmllm.duckdns.org/tcp/8810/p2p/12D3KooWNisnVha2jYj1gqqY5WP82vNQbRhFtBcKzj4XrYmGEn8G";
        assert_eq!(r(dns), dns);
        assert!(r("/ip4/212.132.104.177/tcp/8810").contains("212.132.104.177"));
    }

    #[test]
    fn a_relay_circuit_still_reads_as_a_relay_circuit() {
        let circuit =
            "/ip4/203.0.113.9/udp/8800/quic-v1/p2p/12D3KooWRelay/p2p-circuit/p2p/12D3KooWTarget";
        let out = r(circuit);
        assert!(!out.contains("203.0.113.9"), "{out}");
        assert!(out.contains("/p2p-circuit/"), "{out}");
        assert!(out.contains("12D3KooWRelay"), "{out}");
        assert!(out.contains("12D3KooWTarget"), "{out}");
        assert!(out.contains("quic-v1"), "{out}");
    }

    #[test]
    fn the_same_host_gets_the_same_tag_and_a_different_one_does_not() {
        // This is what keeps "ten cache entries, all one host" distinguishable
        // from "ten different hosts" after redaction.
        let out = r("/ip4/198.51.100.7/tcp/1 /ip4/198.51.100.7/udp/2 /ip4/198.51.100.8/tcp/3");
        let tags: Vec<&str> = out.matches("<public-ip-").map(|_| "").collect();
        assert_eq!(tags.len(), 3);
        let first = out.split_whitespace().next().unwrap();
        let second = out.split_whitespace().nth(1).unwrap();
        let third = out.split_whitespace().nth(2).unwrap();
        let host_of = |s: &str| s.split('/').nth(2).unwrap().to_string();
        assert_eq!(host_of(first), host_of(second));
        assert_ne!(host_of(first), host_of(third));
    }

    #[test]
    fn a_random_salt_makes_the_tag_unrecoverable_across_reports() {
        // IPv4 is 2^32 addresses, so an unsalted digest is a reversible
        // encoding of the host rather than a redaction of it.
        let one = redact_with_salt("/ip4/198.51.100.7/tcp/1", &[1u8; 16]);
        let two = redact_with_salt("/ip4/198.51.100.7/tcp/1", &[2u8; 16]);
        assert_ne!(one, two);
    }

    #[test]
    fn an_address_in_prose_is_redacted_too() {
        // The point of running one pass over the finished report: a section
        // added later, or an error string quoting a dial target, is covered
        // without its author having to know this exists.
        let out = r("dial to 81.241.51.1:8810 failed (peer 198.51.100.7), retrying.");
        assert!(!out.contains("81.241.51.1"), "{out}");
        assert!(!out.contains("198.51.100.7"), "{out}");
        // Punctuation around the address survives, and so does the port.
        assert!(out.contains(":8810 failed"), "{out}");
        assert!(out.contains("),"), "{out}");
        assert!(out.ends_with("retrying."), "{out}");
    }

    #[test]
    fn a_bracketed_ipv6_keeps_its_port() {
        let out = r("[2001:db8::5]:8810");
        assert!(!out.contains("2001:db8::5"), "{out}");
        assert!(out.ends_with("]:8810"), "{out}");
    }

    #[test]
    fn text_with_no_addresses_comes_back_byte_for_byte() {
        let report = "SwarmLLM diagnostics\nversion: 0.3.141-alpha\nnode:    225e6fe7f2b5cd74\n\
                      uptime:  10h58m\n  9684263580c6660f\n  qwen2.5-14b-instruct-q4-k-m: 1/16 shards\n";
        assert_eq!(r(report), report);
    }
}
