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
//! holder, cannot make a peer accept work it would otherwise refuse, and never
//! leaves this machine — a peer is not told that the coordinator was pretending.
//! So the worst a caller can do with it is make their own request slower or
//! fail, which is why it needs no permission beyond the API key every request
//! already carries.
//!
//! It also carries no "at least N segments" knob. How many segments a route has
//! is decided by the priced search over the candidates, not by the candidate set
//! — asking for a segment count would mean overriding the DP's answer rather
//! than its input, and a plan the router did not actually choose is not evidence
//! about routing. Excluding the peers that hold the whole model is the honest
//! way to get a multi-hop route: the search then has to chain partial holders,
//! and the plan it produces is a real one.

use serde::{Deserialize, Serialize};

use crate::error::SwarmError;

pub use swarmllm_types::inference::{PretendLocalHolds, RoutePlanOverride};

/// The wire shape, as it appears on a request body.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwarmRouteRequest {
    /// `"all"` (the default), `"none"`, or an inclusive shard range `"0-3"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pretend_local_holds: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude_nodes: Vec<String>,
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
            Some(s) => Some(parse_pretend_local_holds(s)?),
        };

        let mut exclude_node_prefixes = Vec::with_capacity(self.exclude_nodes.len());
        for raw in &self.exclude_nodes {
            let p = raw.trim().to_ascii_lowercase();
            if p.len() < MIN_PREFIX_HEX || p.len() > MAX_PREFIX_HEX {
                return Err(SwarmError::Validation(format!(
                    "swarm_route.exclude_nodes: '{raw}' is {} characters; give between \
                     {MIN_PREFIX_HEX} and {MAX_PREFIX_HEX} hex characters of a node id \
                     (the peer list prints 16)",
                    p.len()
                )));
            }
            if !p.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(SwarmError::Validation(format!(
                    "swarm_route.exclude_nodes: '{raw}' is not hexadecimal — a node id is \
                     hex, as printed by the peer list and the diagnostics report"
                )));
            }
            exclude_node_prefixes.push(p);
        }

        Ok(RoutePlanOverride {
            pretend_local_holds,
            exclude_node_prefixes,
        })
    }
}

fn parse_pretend_local_holds(s: &str) -> Result<PretendLocalHolds, SwarmError> {
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
                    "swarm_route.pretend_local_holds: '{s}' runs backwards — the range is \
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
        "swarm_route.pretend_local_holds: '{s}' is not one of 'all', 'none', a shard index \
         such as '2', or an inclusive range such as '0-3'"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;

    fn req(holds: Option<&str>, exclude: &[&str]) -> SwarmRouteRequest {
        SwarmRouteRequest {
            pretend_local_holds: holds.map(str::to_string),
            exclude_nodes: exclude.iter().map(|s| s.to_string()).collect(),
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
}
