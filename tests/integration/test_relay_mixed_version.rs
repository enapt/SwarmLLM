//! NETWORKING_PLAN — integration test for the application-level inference relay
//! data path + mixed-version negotiation.
//!
//! Exercises the full A → relay(R) → B round-trip at the wire + crypto level
//! (which the in-crate unit tests don't): the relay never sees plaintext, the
//! target recovers the exact inner message byte-for-byte across two serialize/
//! deserialize hops, and a feature-less (older) peer is never chosen as a relay
//! target. This is the "mixed-version swarm" guard the plan called for, at a
//! level that is deterministic (no libp2p timing/flakiness) yet covers the real
//! sealed-envelope transport a vN ↔ vN pair uses and the vN ↔ vN-1 gate.

use swarmllm::crypto::relay_seal::{open_relayed_message, seal_relayed_message};
use swarmllm::identity::Identity;
use swarmllm::types::{features, ModelId, RemoteGenerateRequest, SamplingParams, SwarmMessage};

fn wire_round_trip(msg: &SwarmMessage) -> SwarmMessage {
    // Simulate one network hop: serialize (send) then deserialize (receive),
    // exactly as the unified codec carries a control SwarmMessage.
    let bytes = serde_json::to_vec(msg).expect("serialize envelope");
    serde_json::from_slice(&bytes).expect("deserialize envelope")
}

#[test]
fn relay_round_trip_relay_blind_target_recovers() {
    let a = Identity::generate(); // origin (NAT'd coordinator)
    let r = Identity::generate(); // relay (the anchor / any reachable peer)
    let b = Identity::generate(); // target (server holding the model)

    let secret_prompt = "the prompt only the two endpoints may read".to_string();
    let request_id = uuid::Uuid::new_v4();
    let inner = SwarmMessage::RemoteGenerateRequest(RemoteGenerateRequest {
        request_id,
        model_id: ModelId("qwen2.5-coder-7b-instruct-q4-k-m".into()),
        layer_range: (0, 28),
        prompt: secret_prompt.clone(),
        sampling: SamplingParams::default(),
        session_id: None,
        sender_peer_bytes: None,
    });

    // A seals the inner request end-to-end for B; only the cleartext routing
    // header (relay_to = B, origin = A) is visible to the relay.
    let env = seal_relayed_message(a.node_id().clone(), b.node_id().clone(), request_id, &inner)
        .expect("seal for target");
    assert_eq!(&env.relay_to, b.node_id());
    assert_eq!(&env.origin, a.node_id());
    // The plaintext prompt must NOT appear in the sealed bytes.
    assert!(
        !env.sealed
            .windows(secret_prompt.len())
            .any(|w| w == secret_prompt.as_bytes()),
        "sealed payload must not contain the plaintext prompt"
    );

    // Hop A -> R over the wire.
    let SwarmMessage::RelayedEnvelope(env_at_relay) =
        wire_round_trip(&SwarmMessage::RelayedEnvelope(env))
    else {
        panic!("expected a RelayedEnvelope at the relay");
    };

    // The relay is a dumb pipe: it CANNOT open the payload (it is not relay_to).
    let r_secret = r.x25519_secret();
    assert!(
        open_relayed_message(&r_secret, &env_at_relay).is_err(),
        "the relay must never be able to read the sealed inner message"
    );

    // Hop R -> B: the relay forwards the envelope verbatim.
    let SwarmMessage::RelayedEnvelope(env_at_target) =
        wire_round_trip(&SwarmMessage::RelayedEnvelope(env_at_relay))
    else {
        panic!("expected a RelayedEnvelope at the target");
    };

    // B opens it and recovers the EXACT inner request.
    let b_secret = b.x25519_secret();
    let recovered = open_relayed_message(&b_secret, &env_at_target).expect("target opens payload");
    match recovered {
        SwarmMessage::RemoteGenerateRequest(got) => {
            assert_eq!(got.request_id, request_id);
            assert_eq!(got.prompt, secret_prompt);
            assert_eq!(got.layer_range, (0, 28));
        }
        other => panic!("recovered wrong message type: {other:?}"),
    }
}

#[test]
fn tampered_relay_header_is_rejected_by_target() {
    // A relay that rewrites the cleartext routing header (e.g. to re-address or
    // spoof the origin) breaks the AAD, so the target rejects it before parsing.
    let a = Identity::generate();
    let b = Identity::generate();
    let evil = Identity::generate();

    let inner = SwarmMessage::CancelInference(swarmllm::types::CancelInference {
        request_id: uuid::Uuid::new_v4(),
    });
    let mut env = seal_relayed_message(
        a.node_id().clone(),
        b.node_id().clone(),
        uuid::Uuid::new_v4(),
        &inner,
    )
    .unwrap();
    env.origin = evil.node_id().clone(); // relay tampers with the header

    let b_secret = b.x25519_secret();
    assert!(open_relayed_message(&b_secret, &env).is_err());
}

#[test]
fn mixed_version_gate_skips_feature_less_peer() {
    // The negotiation invariant a vN ↔ vN-1 swarm relies on: a vN node routes
    // relay traffic only to a peer that advertises the RELAY feature. A vN-1
    // node advertises no features (0) and is skipped — never handed a variant it
    // cannot decode. (The wire parsing that produces `features == 0` for an old
    // peer is covered by `swarmllm-types` version_compat_tests.)
    assert!(
        features::supports(features::ALL, features::RELAY),
        "a current peer is a valid relay target"
    );
    assert!(
        !features::supports(0, features::RELAY),
        "an older, feature-less peer must be skipped"
    );
}
