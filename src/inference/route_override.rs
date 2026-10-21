//! A per-request instruction to plan as if this node held less of a model, or
//! as if some peers were not reachable.
//!
//! # Why this exists
//!
//! The distributed path — a request whose layers are spread over several
//! machines — is the one this project is for, and it is the hardest one to
//! exercise deliberately. Auto-manage converges nodes on holding whole models
//! (that is its job: replication), so a development swarm drifts towards every
//! node being able to answer everything locally, and the multi-hop route that
//! matters stops being taken. `examples/*_sharded_setup.sh` build a split swarm
//! from scratch to get around this, which costs a full model distribution per
//! scenario.
//!
//! This is the cheap version: one request, planned as though the holdings were
//! different, with nothing on disk touched and no node restarted. A harness can
//! sweep the shapes — every layer remote, a named peer excluded, only a slice
//! held locally — against a single live node.
//!
//! # What it deliberately cannot do
//!
//! It only ever makes this node's candidate set SMALLER. It cannot invent a
//! holder and cannot make a peer accept work it would otherwise refuse. It
//! leaves this machine in ONE case: a split handed to its head to lead
//! (#318 (b)) carries it (`DelegatedSplit::route`, only to a head advertising
//! `features::CARRIED_ROUTE`), and the head plans under it read from its own
//! side ([`led_override`]) — again only ever shrinking its candidates, for that
//! request alone. So the worst a caller can do with it is make their own
//! request slower or fail, which is why it needs no permission beyond the API
//! key every request already carries.
//!
//! It also carries no "at least N segments" knob. How many segments a route has
//! is decided by the priced search over the candidates, not by the candidate set
//! — asking for a segment count would mean overriding the DP's answer rather
//! than its input, and a plan the router did not actually choose is not evidence
//! about routing. Excluding the peers that hold the whole model is the honest
//! way to get a multi-hop route: the search then has to chain partial holders,
//! and the plan it produces is a real one.
//!
//! **Restricting a peer (`pretend_peer_holds`) is the same move, one machine
//! over**, and it is what measuring a split needs when every candidate could
//! hold the model: two GPUs that can each run it will never be chained by a
//! search that is free to use one (2026-09-27 — no exclusion could produce the
//! split to measure). Ours restricted to shards 0-3 and the peer to 4-7, the
//! search still chooses — over an input where the split is the only route.

use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::types::NodeId;

pub use swarmllm_types::inference::{PretendLocalHolds, RoutePlanOverride};

/// The wire shape, as it appears on a request body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmRouteRequest {
    /// `"all"` (the default), `"none"`, or an inclusive shard range `"0-3"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pretend_local_holds: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude_nodes: Vec<String>,
    /// Per peer, a node-id prefix mapped to what it may be asked for: `"none"`,
    /// a shard index, or an inclusive range — the grammar of
    /// `pretend_local_holds`, applied to someone else. `{"bf7b": "4-7"}` plans as
    /// though that peer held shards 4-7 only.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub pretend_peer_holds: std::collections::BTreeMap<String, String>,
}

/// Longest node-id prefix a caller may give, in hex characters.
///
/// A `NodeId` is 32 bytes, so 64 hex characters is the whole thing. Anything
/// longer is a typo rather than a more precise answer, and saying so beats
/// silently matching nothing.
const MAX_PREFIX_HEX: usize = 64;

/// Shortest prefix that still names one machine rather than a class of them.
///
/// Four hex characters is 16 bits. The diagnostics report and the peer list both
/// print 16 characters, so anyone copying an id off a screen has far more than
/// this; the floor exists to catch an empty or one-character entry that would
/// quietly exclude a large part of the swarm.
const MIN_PREFIX_HEX: usize = 4;

impl SwarmRouteRequest {
    /// Validate and parse, or say exactly what was wrong with it.
    ///
    /// Every failure here is the caller's input, so they are all
    /// [`SwarmError::Validation`] → 400. Returning 500 for a mistyped range
    /// would tell a retry-on-5xx client to send it again forever.
    pub fn parse(&self) -> Result<RoutePlanOverride, SwarmError> {
        let pretend_local_holds = match self.pretend_local_holds.as_deref() {
            None => None,
            Some(s) => Some(parse_holdings(s, "pretend_local_holds")?),
        };

        let exclude_node_prefixes = self
            .exclude_nodes
            .iter()
            .map(|raw| parse_node_prefix(raw, "exclude_nodes"))
            .collect::<Result<Vec<_>, _>>()?;

        let mut peer_holds = Vec::with_capacity(self.pretend_peer_holds.len());
        for (raw_node, raw_holds) in &self.pretend_peer_holds {
            peer_holds.push((
                parse_node_prefix(raw_node, "pretend_peer_holds")?,
                parse_holdings(raw_holds, "pretend_peer_holds")?,
            ));
        }

        Ok(RoutePlanOverride {
            pretend_local_holds,
            exclude_node_prefixes,
            peer_holds,
            // A client cannot ask for this: it is the delegate's planning
            // instruction, set only by `handle_delegated_split`.
            lead_here: false,
        })
    }
}

/// A node-id prefix as the peer list prints it, or the reason it is not one.
fn parse_node_prefix(raw: &str, field: &str) -> Result<String, SwarmError> {
    let p = raw.trim().to_ascii_lowercase();
    if p.len() < MIN_PREFIX_HEX || p.len() > MAX_PREFIX_HEX {
        return Err(SwarmError::Validation(format!(
            "swarm_route.{field}: '{raw}' is {} characters; give between \
             {MIN_PREFIX_HEX} and {MAX_PREFIX_HEX} hex characters of a node id \
             (the peer list prints 16)",
            p.len()
        )));
    }
    if !p.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(SwarmError::Validation(format!(
            "swarm_route.{field}: '{raw}' is not hexadecimal — a node id is \
             hex, as printed by the peer list and the diagnostics report"
        )));
    }
    Ok(p)
}

fn parse_holdings(s: &str, field: &str) -> Result<PretendLocalHolds, SwarmError> {
    let t = s.trim().to_ascii_lowercase();
    match t.as_str() {
        "all" | "everything" => return Ok(PretendLocalHolds::Everything),
        "none" | "nothing" => return Ok(PretendLocalHolds::Nothing),
        _ => {}
    }
    // An inclusive range, spelled as `inference.shard_range` spells it.
    if let Some((a, b)) = t.split_once('-') {
        if let (Ok(start), Ok(end)) = (a.trim().parse::<u32>(), b.trim().parse::<u32>()) {
            if start > end {
                return Err(SwarmError::Validation(format!(
                    "swarm_route.{field}: '{s}' runs backwards — the range is \
                     inclusive and written low-high, as in '0-3'"
                )));
            }
            return Ok(PretendLocalHolds::Shards(start, end));
        }
    }
    // A single shard is a range of one, which is how `shard_range` reads it too.
    if let Ok(only) = t.parse::<u32>() {
        return Ok(PretendLocalHolds::Shards(only, only));
    }
    Err(SwarmError::Validation(format!(
        "swarm_route.{field}: '{s}' is not one of 'all', 'none', a shard index \
         such as '2', or an inclusive range such as '0-3'"
    )))
}

/// Most entries a head takes in each list of a carried route. The API body's
/// size bounds what a client sends; this bounds what a peer makes the head
/// match every candidate against.
const MAX_CARRIED_ENTRIES: usize = 64;

/// The instruction a request is handed to its head with (#318 (b)): what the
/// requester was told, unchanged — the head reads it from its own side
/// ([`led_override`]).
pub(crate) fn carried(o: &RoutePlanOverride) -> swarmllm_types::CarriedRoute {
    swarmllm_types::CarriedRoute {
        requester_holds: o.pretend_local_holds,
        exclude_node_prefixes: o.exclude_node_prefixes.clone(),
        peer_holds: o.peer_holds.clone(),
    }
}

/// The plan instruction a head leads a handed-over split under (#318 (b)):
/// lead it here, and keep to what the requester asked, read from here.
///
/// Every part only SHRINKS the head's candidates, as it shrank the
/// requester's:
/// - an entry of `peer_holds` naming this node (the first, as everywhere) is
///   what this node pretends to hold. The entry stays in `peer_holds` too: a
///   prefix may name other machines as well, and dropping it would lift their
///   restriction;
/// - `requester_holds` — what the requester pretended about ITSELF — applies
///   to `requester` (the AUTHENTICATED sender), ahead of every other entry, so
///   it decides for the requester as it did there;
/// - exclusions go on as they are.
///
/// Refused — the head then declines the hand-off, and the requester leads the
/// plan it already has — where the route is malformed, excludes this node,
/// or leaves it nothing to hold: the requester planned none of that.
pub(crate) fn led_override(
    route: Option<&swarmllm_types::CarriedRoute>,
    me: &NodeId,
    requester: &NodeId,
) -> Result<RoutePlanOverride, &'static str> {
    let Some(route) = route else {
        return Ok(RoutePlanOverride::lead_here());
    };
    if route.exclude_node_prefixes.len() > MAX_CARRIED_ENTRIES
        || route.peer_holds.len() > MAX_CARRIED_ENTRIES
    {
        return Err("the route names more nodes than a head takes");
    }
    let well_formed = |p: &String| {
        (MIN_PREFIX_HEX..=MAX_PREFIX_HEX).contains(&p.len())
            && p.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
    };
    if !route.exclude_node_prefixes.iter().all(well_formed)
        || !route.peer_holds.iter().all(|(p, _)| well_formed(p))
    {
        return Err("the route names a node by something other than a hex prefix");
    }
    // The request API refuses an inverted range; the wire does not, and one
    // naming this node would read as "holds nothing" past the check below.
    let inverted = |h: &PretendLocalHolds| matches!(h, PretendLocalHolds::Shards(a, b) if a > b);
    if route.requester_holds.as_ref().is_some_and(inverted)
        || route.peer_holds.iter().any(|(_, h)| inverted(h))
    {
        return Err("the route names a range of shards that ends before it starts");
    }
    let mine = hex::encode(me.0);
    if route
        .exclude_node_prefixes
        .iter()
        .any(|p| mine.starts_with(p.as_str()))
    {
        return Err("the route excludes this node");
    }
    let pretend_local_holds = route
        .peer_holds
        .iter()
        .find(|(p, _)| mine.starts_with(p.as_str()))
        .map(|(_, holds)| *holds);
    if pretend_local_holds == Some(PretendLocalHolds::Nothing) {
        return Err("the route leaves this node none of the model to lead with");
    }
    let mut peer_holds = Vec::with_capacity(route.peer_holds.len() + 1);
    peer_holds.push((
        hex::encode(requester.0),
        route
            .requester_holds
            .unwrap_or(PretendLocalHolds::Everything),
    ));
    peer_holds.extend(route.peer_holds.iter().cloned());
    Ok(RoutePlanOverride {
        pretend_local_holds,
        exclude_node_prefixes: route.exclude_node_prefixes.clone(),
        peer_holds,
        lead_here: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;

    fn req(holds: Option<&str>, exclude: &[&str]) -> SwarmRouteRequest {
        SwarmRouteRequest {
            pretend_local_holds: holds.map(str::to_string),
            exclude_nodes: exclude.iter().map(|s| s.to_string()).collect(),
            pretend_peer_holds: Default::default(),
        }
    }

    #[test]
    fn holdings_spellings_all_parse() {
        assert_eq!(
            req(Some("none"), &[]).parse().unwrap().pretend_local_holds,
            Some(PretendLocalHolds::Nothing)
        );
        assert_eq!(
            req(Some("ALL"), &[]).parse().unwrap().pretend_local_holds,
            Some(PretendLocalHolds::Everything)
        );
        assert_eq!(
            req(Some(" 0-3 "), &[]).parse().unwrap().pretend_local_holds,
            Some(PretendLocalHolds::Shards(0, 3))
        );
        // A single shard is a range of one, matching `inference.shard_range`.
        assert_eq!(
            req(Some("2"), &[]).parse().unwrap().pretend_local_holds,
            Some(PretendLocalHolds::Shards(2, 2))
        );
        // Saying nothing is not the same as saying "all": it must stay None so
        // the scheduler can skip the whole mechanism.
        assert!(req(None, &[]).parse().unwrap().is_noop());
    }

    /// Every rejection is the caller's input, so every one is a 400. A 500 here
    /// would tell a retry-on-5xx client to re-send a request that can never
    /// succeed — the same trap `classify_error`'s contract exists to close.
    #[test]
    fn a_mistyped_override_is_the_callers_mistake_not_a_server_fault() {
        for bad in ["sometimes", "3-1", "", "0-"] {
            let err = req(Some(bad), &[]).parse().unwrap_err();
            assert!(
                matches!(err, SwarmError::Validation(_)),
                "'{bad}' should be a 400, got {err:?}"
            );
        }
        for bad in ["zz", "abc", "", "  "] {
            let err = req(None, &[bad]).parse().unwrap_err();
            assert!(
                matches!(err, SwarmError::Validation(_)),
                "exclude '{bad}' should be a 400, got {err:?}"
            );
        }
    }

    /// A backwards range is rejected rather than quietly normalised. Silently
    /// swapping the ends would make a scenario file mean something other than
    /// what it says, and the whole point of this knob is that the shape under
    /// test is the shape that was asked for.
    #[test]
    fn a_backwards_range_is_refused_not_reordered() {
        let err = req(Some("5-2"), &[]).parse().unwrap_err();
        assert!(err.to_string().contains("runs backwards"), "{err}");
    }

    #[test]
    fn pretending_to_hold_nothing_releases_every_shard() {
        let o = req(Some("none"), &[]).parse().unwrap();
        assert!(!o.is_noop());
        for i in 0..16 {
            assert!(!o.local_holds(i), "shard {i} should be released");
        }
    }

    #[test]
    fn a_slice_keeps_only_its_own_shards_inclusive_at_both_ends() {
        let o = req(Some("2-4"), &[]).parse().unwrap();
        assert!(!o.local_holds(1));
        assert!(o.local_holds(2));
        assert!(o.local_holds(4), "the range is inclusive at the top");
        assert!(!o.local_holds(5));
    }

    /// **A peer restricted to a range is asked for that range only** — the
    /// peer-side sibling of `pretend_local_holds`, same grammar, same 400s. It is
    /// how a benchmark makes a two-machine split the ONLY plan: the search still
    /// makes it, over a smaller input (2026-09-27 — with two cards that could each
    /// hold the model, no other override could produce a split to measure).
    #[test]
    fn a_peer_can_be_restricted_to_a_range_of_shards() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0xbf;
        bytes[1] = 0x7b;
        let peer = NodeId(bytes);
        let other = NodeId([0x11u8; 32]);
        let mut r = req(Some("0-3"), &[]);
        r.pretend_peer_holds.insert("BF7B".into(), " 4-7 ".into());
        let o = r.parse().unwrap();
        assert!(!o.is_noop());
        assert!(!o.peer_holds_shard(&peer, 3));
        assert!(o.peer_holds_shard(&peer, 4));
        assert!(o.peer_holds_shard(&peer, 7), "inclusive at the top");
        assert!(!o.peer_holds_shard(&peer, 8));
        assert!(
            o.peer_holds_shard(&other, 0) && o.peer_holds_shard(&other, 15),
            "a peer the override does not name is untouched"
        );
        // Restricting a peer says nothing about us.
        assert!(o.local_holds(3) && !o.local_holds(4));

        // Alone, it is still an instruction — the scheduler must not skip it.
        let mut only = req(None, &[]);
        only.pretend_peer_holds.insert("bf7b".into(), "none".into());
        let o = only.parse().unwrap();
        assert!(!o.is_noop() && !o.peer_holds_shard(&peer, 0));

        for (node, holds) in [
            ("zz", "4-7"),
            ("bf7b", "7-4"),
            ("bf7b", "sometimes"),
            ("", "1"),
        ] {
            let mut bad = req(None, &[]);
            bad.pretend_peer_holds.insert(node.into(), holds.into());
            let err = bad.parse().unwrap_err();
            assert!(
                matches!(err, SwarmError::Validation(_))
                    && err.to_string().contains("pretend_peer_holds"),
                "'{node}': '{holds}' should be a 400 naming the field, got {err:?}"
            );
        }
    }

    /// The prefix is matched against the same hex spelling the peer list and
    /// diagnostics report print, so a node can be excluded by copying its id
    /// off a screen. Case is not part of the identity.
    #[test]
    fn a_peer_is_excluded_by_the_prefix_the_ui_shows() {
        let mut bytes = [0u8; 32];
        bytes[0] = 0xbf;
        bytes[1] = 0x7b;
        bytes[2] = 0x32;
        bytes[3] = 0x63;
        let peer = NodeId(bytes);
        let other = NodeId([0x11u8; 32]);

        let o = req(None, &["BF7B3263"]).parse().unwrap();
        assert!(o.excludes_peer(&peer));
        assert!(!o.excludes_peer(&other));

        // No exclusions means no work and no exclusion.
        let none = req(None, &[]).parse().unwrap();
        assert!(!none.excludes_peer(&peer));
    }

    fn node(first: u8) -> NodeId {
        let mut bytes = [0x11u8; 32];
        bytes[0] = first;
        bytes[1] = 0x7b;
        NodeId(bytes)
    }

    /// The route a requester planned under reaches its head read from the
    /// head's side (#318 (b)): the head holds what the requester pretended
    /// it held, the requester holds what it pretended about itself, and every
    /// other machine is held to what it was held to there.
    #[test]
    fn a_head_plans_a_carried_route_from_its_own_side() {
        let (head, requester, tail, other) = (node(0xaa), node(0xbb), node(0xcc), node(0xdd));
        let mut asked = SwarmRouteRequest {
            pretend_local_holds: Some("none".into()),
            exclude_nodes: vec!["dd7b".into()],
            pretend_peer_holds: Default::default(),
        };
        asked.pretend_peer_holds.insert("aa7b".into(), "0-3".into());
        asked.pretend_peer_holds.insert("cc7b".into(), "4-7".into());
        let at_requester = asked.parse().unwrap();
        let o = led_override(Some(&carried(&at_requester)), &head, &requester).unwrap();
        assert!(o.lead_here && !o.lets_peer_lead());
        assert!(o.local_holds(3) && !o.local_holds(4), "the head holds 0-3");
        assert!(
            !o.peer_holds_shard(&requester, 0),
            "the requester pretended to hold nothing, and still does"
        );
        assert!(o.peer_holds_shard(&tail, 5) && !o.peer_holds_shard(&tail, 2));
        assert!(o.excludes_peer(&other));
        // Nothing carried: lead here, change nothing else.
        assert_eq!(
            led_override(None, &head, &requester).unwrap(),
            RoutePlanOverride::lead_here()
        );
    }

    /// A carried route only shrinks: an entry naming the head AND other
    /// machines still holds the others to it, and a requester that said
    /// nothing about itself is planned as it really is.
    #[test]
    fn a_carried_route_never_lifts_a_restriction() {
        let (head, requester, twin) = (node(0xaa), node(0xbb), node(0xaa));
        let mut twin_bytes = twin.0;
        twin_bytes[31] = 0x99;
        let twin = NodeId(twin_bytes);
        let route = swarmllm_types::CarriedRoute {
            requester_holds: None,
            exclude_node_prefixes: vec![],
            peer_holds: vec![("aa7b".into(), PretendLocalHolds::Shards(0, 1))],
        };
        let o = led_override(Some(&route), &head, &requester).unwrap();
        assert!(o.local_holds(1) && !o.local_holds(2));
        assert!(
            !o.peer_holds_shard(&twin, 2),
            "another machine under the same prefix keeps its restriction"
        );
        assert!(o.peer_holds_shard(&requester, 9));
    }

    /// What the requester could not have planned is refused, and the head
    /// declines the hand-off: a route excluding the head, leaving it nothing,
    /// naming nodes by anything but a hex prefix, or naming too many.
    #[test]
    fn a_route_the_requester_could_not_have_planned_is_refused() {
        let (head, requester) = (node(0xaa), node(0xbb));
        let refuse = |route: swarmllm_types::CarriedRoute| {
            led_override(Some(&route), &head, &requester).is_err()
        };
        assert!(refuse(swarmllm_types::CarriedRoute {
            exclude_node_prefixes: vec!["aa7b".into()],
            ..Default::default()
        }));
        assert!(refuse(swarmllm_types::CarriedRoute {
            peer_holds: vec![("aa7b".into(), PretendLocalHolds::Nothing)],
            ..Default::default()
        }));
        assert!(refuse(swarmllm_types::CarriedRoute {
            exclude_node_prefixes: vec!["AA7B".into()],
            ..Default::default()
        }));
        assert!(refuse(swarmllm_types::CarriedRoute {
            exclude_node_prefixes: vec!["cc".into()],
            ..Default::default()
        }));
        assert!(refuse(swarmllm_types::CarriedRoute {
            exclude_node_prefixes: vec!["cc7b".into(); MAX_CARRIED_ENTRIES + 1],
            ..Default::default()
        }));
        assert!(!refuse(swarmllm_types::CarriedRoute {
            exclude_node_prefixes: vec!["cc7b".into(); MAX_CARRIED_ENTRIES],
            ..Default::default()
        }));
        // An inverted range naming the head read as "holds nothing" past the
        // check for `Nothing` (security review, 2026-10-21).
        assert!(refuse(swarmllm_types::CarriedRoute {
            peer_holds: vec![("aa7b".into(), PretendLocalHolds::Shards(3, 1))],
            ..Default::default()
        }));
        assert!(refuse(swarmllm_types::CarriedRoute {
            requester_holds: Some(PretendLocalHolds::Shards(3, 1)),
            ..Default::default()
        }));
        assert!(!refuse(swarmllm_types::CarriedRoute {
            peer_holds: vec![("aa7b".into(), PretendLocalHolds::Shards(1, 1))],
            ..Default::default()
        }));
    }
}
