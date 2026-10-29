//! The in-engine drafter (`SplitModel::draft_after`) on a tiny whole model on
//! the processor: its incremental calls — cache cut back to what a check
//! confirmed, only the new tokens read — must guess exactly what the model
//! guesses reading the whole sequence from nothing, over the round shapes the
//! coordinator produces (`pipeline::engine_drafter`).

use super::super::token_embedding::TokenEmbedding;
use super::super::*;
use super::common::*;
use candle_core::{Device, Tensor};

const VOCAB: usize = 256;

/// A whole model — embedding, 3 dense layers, final norm, output head — with
/// random weights and logits spread wide enough that a batched and a
/// one-position pass agree on every argmax.
fn whole_model() -> SplitModel {
    let hidden = 128;
    let mut m = make_test_split_model(3, hidden);
    let table = Tensor::randn(0f32, 1.0, (VOCAB, hidden), &Device::Cpu).unwrap();
    m.tok_embeddings = Some(TokenEmbedding::Dense(candle_nn::Embedding::new(
        table, hidden,
    )));
    m.norm = Some(make_rms_norm_dim(hidden, &Device::Cpu));
    let w = (Tensor::randn(0f32, 1.0, (VOCAB, hidden), &Device::Cpu).unwrap() * 0.5).unwrap();
    m.output = Some(
        QMatMul::from_qtensor(
            candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32)
                .unwrap(),
        )
        .unwrap(),
    );
    m.total_layers = 3;
    m.layer_end = 3;
    m.kv_model_key = "0-3-3".into();
    m
}

fn greedy() -> crate::types::SamplingParams {
    crate::types::SamplingParams {
        temperature: 0.0,
        ..Default::default()
    }
}

/// What the model guesses reading `seq` from an empty cache.
fn fresh(model: &mut SplitModel, seq: &[u32], gamma: usize) -> Vec<u32> {
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    model
        .draft_after(&kv, "fresh", 0, seq, gamma, &greedy(), &[], None, 0, None)
        .unwrap()
}

/// No tree to grow on.
static NO_SEED: crate::inference::draft_tree::GrownTree = crate::inference::draft_tree::GrownTree {
    tokens: Vec::new(),
    parents: Vec::new(),
    path_logp: Vec::new(),
};

/// `rows` guesses grown from nothing, `per_step` a forward, with no time limit.
fn grow(rows: usize, per_step: usize) -> TreeGrowth<'static> {
    TreeGrowth {
        seed: &NO_SEED,
        rows,
        per_step,
        grow_for: None,
    }
}

fn cache_len(model: &SplitModel, kv: &KvCacheStore, req: &str) -> usize {
    kv.get_or_create_keyed(&KvCacheStore::cache_key(model.kv_model_key(), req), 3)
        .layers[0]
        .as_ref()
        .map_or(0, |l| l.current_seq_len())
}

/// A forward that skips layers reports only the processor layers it RUNS to
/// the decode-thread calibration (`SplitModel::cpu_layers_run`, read where
/// every forward picks its pool). A card guess skips all of them; reported as
/// a full processor token (7-13 ms against ~150 for a real one) it taught the
/// calibration that one thread was as fast as four, and a 14B split's every
/// later step ran on one (2026-10-28).
#[test]
fn a_skipped_layer_is_not_reported_as_processor_work() {
    let model = make_test_split_model(4, 128);
    assert_eq!(model.cpu_layers_run(None), 4);
    assert_eq!(model.cpu_layers_run(Some(&[false, true, true, false])), 2);
    assert_eq!(model.cpu_layers_run(Some(&[true; 4])), 0, "a card guess");
    // A mask shorter than the model skips nothing past its end.
    assert_eq!(model.cpu_layers_run(Some(&[true])), 3);
}

#[test]
fn incremental_drafting_guesses_what_a_fresh_read_guesses() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12];

    // Round 1: the prompt and the first reply token, four guesses.
    let mut seq = prompt.clone();
    seq.push(5);
    let d1 = model
        .draft_after(&kv, "r", 0, &seq, 4, &greedy(), &[], None, 0, None)
        .unwrap();
    assert_eq!(d1, fresh(&mut model, &seq, 4));
    assert_eq!(cache_len(&model, &kv, "r"), seq.len() + 3);

    // The check kept two and sampled something else third: the cache keeps
    // what it read of those two, and the next call reads only the sample.
    let valid = seq.len() + 2;
    seq.extend_from_slice(&d1[..2]);
    let sample = (d1[2] + 1) % VOCAB as u32;
    seq.push(sample);
    kv.truncate_request_to(model.kv_model_key(), "r", valid)
        .unwrap();
    let d2 = model
        .draft_after(
            &kv,
            "r",
            valid,
            &seq[valid..],
            3,
            &greedy(),
            &[],
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(d2, fresh(&mut model, &seq, 3));

    // Every guess kept: the last one was never read, so it is read now with
    // the check's sample after it.
    let valid = seq.len() + 2;
    seq.extend_from_slice(&d2);
    seq.push((d2[2] + 7) % VOCAB as u32);
    kv.truncate_request_to(model.kv_model_key(), "r", valid)
        .unwrap();
    assert_eq!(seq.len() - valid, 2, "the unread last guess and the sample");
    let d3 = model
        .draft_after(
            &kv,
            "r",
            valid,
            &seq[valid..],
            2,
            &greedy(),
            &[],
            None,
            0,
            None,
        )
        .unwrap();
    assert_eq!(d3, fresh(&mut model, &seq, 2));
    assert_eq!(cache_len(&model, &kv, "r"), seq.len() + 1);
}

/// An unsure guess ends the round after itself: a floor no guess can reach
/// gives exactly one, and the cache then holds what was read and nothing guessed.
#[test]
fn a_guess_below_the_confidence_floor_ends_the_round_after_itself() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let seq: Vec<u32> = vec![4, 8, 15, 16, 23, 42];
    let one = model
        .draft_after(&kv, "c", 0, &seq, 5, &greedy(), &[], None, 0, Some(1.01))
        .unwrap();
    assert_eq!(one.len(), 1);
    assert_eq!(one, fresh(&mut model, &seq, 1));
    assert_eq!(cache_len(&model, &kv, "c"), seq.len());
    let kv2 = KvCacheStore::new(std::time::Duration::from_secs(60));
    let all = model
        .draft_after(&kv2, "d", 0, &seq, 5, &greedy(), &[], None, 0, Some(0.0))
        .unwrap();
    assert_eq!(all.len(), 5, "a floor of 0 stops nothing");
}

#[test]
fn with_shared_noise_a_guess_is_the_draw_plain_sampling_makes() {
    // At T > 0 a guess is drawn with the noise of the position it will occupy,
    // so it equals what sampling that position with the same seed draws.
    let mut model = whole_model();
    let s = crate::types::SamplingParams {
        temperature: 0.9,
        top_k: 0,
        top_p: 1.0,
        ..Default::default()
    };
    let noise = crate::inference::coupled_noise::CoupledNoise::new(0xD2AF7);
    let seq: Vec<u32> = vec![9, 8, 7, 6, 5];
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let guesses = model
        .draft_after(&kv, "n", 0, &seq, 3, &s, &[], Some(&noise), 0, None)
        .unwrap();

    let kv2 = KvCacheStore::new(std::time::Duration::from_secs(60));
    let mut logits = model
        .forward(&model.tensor_from_ids(&seq).unwrap(), 0, &kv2, "p")
        .unwrap();
    let mut drawn = vec![];
    let mut ctx = crate::inference::sampling::SamplingContext::new(0);
    for j in 0..3 {
        let mut row: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        let t = crate::inference::sampling::sample_token_coupled(
            &mut row,
            &s,
            &drawn,
            &mut ctx,
            &noise,
            (seq.len() + j) as u64,
        );
        drawn.push(t);
        logits = model
            .forward(&model.token_tensor(t).unwrap(), seq.len() + j, &kv2, "p")
            .unwrap();
    }
    assert_eq!(guesses, drawn);
}

/// A call that continues a context this worker no longer holds is refused —
/// the entry expired, or the worker was replaced — never read on top of an
/// empty or shorter cache, which would guess from a context never written and
/// fail nothing.
#[test]
fn a_call_continuing_a_context_the_cache_no_longer_holds_is_refused() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let seq: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let d1 = model
        .draft_after(&kv, "g", 0, &seq, 3, &greedy(), &[], None, 0, None)
        .unwrap();
    // The check kept one guess and sampled the next token itself.
    let valid = seq.len() + 1;
    let append = [d1[1] ^ 1];

    kv.truncate_request_to(model.kv_model_key(), "g", 4)
        .unwrap();
    let shorter = model.draft_after(&kv, "g", valid, &append, 2, &greedy(), &[], None, 0, None);
    assert!(matches!(
        shorter,
        Err(crate::error::SwarmError::ServiceUnavailable(_))
    ));

    kv.clear_request(model.kv_model_key(), "g");
    let gone = model.draft_after(&kv, "g", valid, &append, 2, &greedy(), &[], None, 0, None);
    assert!(matches!(
        gone,
        Err(crate::error::SwarmError::ServiceUnavailable(_))
    ));
}

/// A first read starts from nothing, whatever an earlier call that failed
/// part-way left in the request's cache.
#[test]
fn a_first_read_starts_from_nothing_whatever_an_earlier_call_left() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    model
        .draft_after(
            &kv,
            "s",
            0,
            &[1, 2, 3, 4, 5],
            4,
            &greedy(),
            &[],
            None,
            0,
            None,
        )
        .unwrap();
    let seq: Vec<u32> = vec![9, 8, 7, 6, 200, 17];
    let again = model
        .draft_after(&kv, "s", 0, &seq, 3, &greedy(), &[], None, 0, None)
        .unwrap();
    assert_eq!(again, fresh(&mut model, &seq, 3));
    assert_eq!(cache_len(&model, &kv, "s"), seq.len() + 2);
}

/// The card's loop — each token chosen where the logits are, the next step fed
/// from there, every guess read back at once — guesses what the host loop
/// guesses, and cuts after the same unsure guess. It is ordinary tensor code,
/// so it runs here on the processor as it would on a card.
#[test]
fn guesses_chosen_where_the_logits_are_match_the_host_loop() {
    let mut model = whole_model();
    let seq: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    for floor in [
        None,
        Some(1.01f32),
        Some(0.0),
        Some(0.3),
        Some(0.6),
        Some(0.9),
    ] {
        let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
        let host = model
            .draft_after(&kv, "h", 0, &seq, 5, &greedy(), &[], None, 0, floor)
            .unwrap();
        let kv2 = KvCacheStore::new(std::time::Duration::from_secs(60));
        let logits = model
            .forward(&model.tensor_from_ids(&seq).unwrap(), 0, &kv2, "c")
            .unwrap();
        let card = model
            .argmax_guesses_on_card(&kv2, "c", seq.len(), logits, 5, 0.0, floor, None)
            .unwrap();
        assert_eq!(card, host, "floor {floor:?}");
    }
}

/// A read-ahead asks the card path for NO guesses — the prompt read is the
/// whole call — and must get none back, as the host loop does, not an error:
/// every read-ahead on a card failed with "stack expects at least one tensor"
/// (review of e437b1e0, seen on the 2026-09-30 rigs) and the first round then
/// read the prompt again.
#[test]
fn a_read_ahead_asks_the_card_path_for_nothing_and_gets_nothing() {
    let mut model = whole_model();
    let seq: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let logits = model
        .forward(&model.tensor_from_ids(&seq).unwrap(), 0, &kv, "r")
        .unwrap();
    let guesses = model
        .argmax_guesses_on_card(&kv, "r", seq.len(), logits, 0, 0.0, Some(0.4), None)
        .unwrap();
    assert!(guesses.is_empty());
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "r"),
        seq.len(),
        "the prompt stays in the cache for the first round"
    );
}

/// A tree of guesses (#134, `draft_tree_after`): every guess is read along its
/// own path, so the followers it is given are the ones the model guesses after
/// that path read from nothing; its likeliest path is laid out first and is
/// the chain `draft_after` guesses; ratings never rise along a path; and the
/// cache keeps the context alone.
#[test]
fn a_grown_tree_guesses_after_each_path_what_a_fresh_read_guesses() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let grown = model
        .draft_tree_after(&kv, "t", 0, &prompt, grow(24, 4), &greedy(), &[], None, 0)
        .unwrap();
    assert_eq!(grown.tokens.len(), 24);
    assert_eq!(grown.parents.len(), 24);
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "t"),
        prompt.len()
    );

    let path_to = |row: usize| -> Vec<u32> {
        let mut path = vec![grown.tokens[row]];
        let mut at = grown.parents[row];
        while let Some(p) = at {
            path.push(grown.tokens[usize::from(p)]);
            at = grown.parents[usize::from(p)];
        }
        path.reverse();
        path
    };
    // The likeliest path: row 0, then each row the one before's child.
    let trunk = 1
        + (1..24)
            .take_while(|&r| grown.parents[r] == Some(r as u16 - 1))
            .count();
    assert!(trunk >= 2, "a trunk of {trunk}");
    assert_eq!(
        &grown.tokens[..trunk],
        &fresh(&mut model, &prompt, trunk)[..],
        "the trunk is the greedy chain"
    );
    for row in 0..24 {
        if let Some(p) = grown.parents[row] {
            assert!(grown.path_logp[row] <= grown.path_logp[usize::from(p)] + 1e-6);
        }
        // The likeliest of the followers it was given is what a fresh read of
        // its path guesses next.
        let best_child = (0..24)
            .filter(|&c| grown.parents[c] == Some(row as u16))
            .max_by(|&a, &b| grown.path_logp[a].total_cmp(&grown.path_logp[b]));
        if let Some(c) = best_child {
            let mut seq = prompt.clone();
            seq.extend(path_to(row));
            assert_eq!(
                grown.tokens[c],
                fresh(&mut model, &seq, 1)[0],
                "row {row}'s likeliest follower"
            );
        }
    }
    // The same call again, from the cache it left, grows the same tree.
    let again = model
        .draft_tree_after(
            &kv,
            "t",
            prompt.len() - 1,
            &prompt[prompt.len() - 1..],
            grow(24, 4),
            &greedy(),
            &[],
            None,
            0,
        )
        .unwrap();
    assert_eq!(
        (&again.tokens, &again.parents),
        (&grown.tokens, &grown.parents)
    );
    // The last token read alone rather than in the prompt's batch: the same
    // ratings to rounding.
    for (a, b) in again.path_logp.iter().zip(&grown.path_logp) {
        assert!((a - b).abs() < 1e-4, "{a} vs {b}");
    }
}

/// Every row's path from the last token read, as a set: two trees holding
/// the same guesses in another order are the same tree.
fn paths_of(
    tree: &crate::inference::draft_tree::GrownTree,
) -> std::collections::BTreeSet<Vec<u32>> {
    (0..tree.tokens.len())
        .map(|row| {
            let mut path = vec![tree.tokens[row]];
            let mut at = tree.parents[row];
            while let Some(p) = at {
                path.push(tree.tokens[usize::from(p)]);
                at = tree.parents[usize::from(p)];
            }
            path.reverse();
            path
        })
        .collect()
}

/// A call growing on the tree an earlier call grew (the round loop's growth
/// beside a check, `pipeline::dsd_tree`) goes on where that one stopped: six
/// guesses, then six more on them, are the twelve one call grows — none of
/// the seed's guesses is queued again, and none it left out is lost.
#[test]
fn a_tree_grown_on_an_earlier_one_is_the_tree_one_call_grows() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let last = prompt.len() - 1;
    let whole = model
        .draft_tree_after(&kv, "s", 0, &prompt, grow(12, 1), &greedy(), &[], None, 0)
        .unwrap();
    let first = model
        .draft_tree_after(
            &kv,
            "s",
            last,
            &prompt[last..],
            grow(6, 1),
            &greedy(),
            &[],
            None,
            0,
        )
        .unwrap();
    let more = model
        .draft_tree_after(
            &kv,
            "s",
            last,
            &prompt[last..],
            TreeGrowth {
                seed: &first,
                rows: 6,
                per_step: 1,
                grow_for: None,
            },
            &greedy(),
            &[],
            None,
            0,
        )
        .unwrap();
    assert_eq!(more.tokens.len(), 12);
    assert_eq!(paths_of(&more), paths_of(&whole));
    assert!(paths_of(&first).is_subset(&paths_of(&more)));
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "s"),
        prompt.len(),
        "the cache holds the context alone"
    );
    // A seed whose row names a later row as its parent is refused, and the
    // cache is left as it was.
    let mut backwards = first.clone();
    backwards.parents[0] = Some(1);
    let refused = model.draft_tree_after(
        &kv,
        "s",
        last,
        &prompt[last..],
        TreeGrowth {
            seed: &backwards,
            rows: 6,
            per_step: 1,
            grow_for: None,
        },
        &greedy(),
        &[],
        None,
        0,
    );
    assert!(matches!(
        refused,
        Err(crate::error::SwarmError::Validation(_))
    ));
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "s"),
        prompt.len()
    );
}

/// Past the time it was given, a call adds the guesses already rated and
/// reads nothing more: with no time at all, one step of the root's likeliest
/// followers.
#[test]
fn a_call_out_of_time_adds_one_step_it_does_not_read() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let grown = model
        .draft_tree_after(
            &kv,
            "o",
            0,
            &prompt,
            TreeGrowth {
                grow_for: Some(std::time::Duration::ZERO),
                ..grow(24, 4)
            },
            &greedy(),
            &[],
            None,
            0,
        )
        .unwrap();
    assert_eq!(grown.tokens.len(), 4);
    assert!(grown.parents.iter().all(Option::is_none));
    assert_eq!(grown.tokens[0], fresh(&mut model, &prompt, 1)[0]);
}

/// Row `row`'s path in a tree that flows, from its row 0.
fn flow_path(tree: &crate::inference::draft_tree::FlowTree, row: usize) -> Vec<u32> {
    let mut path = vec![tree.tokens()[row]];
    let mut at = tree.tree().parent(row);
    while let Some(p) = at {
        path.push(tree.tokens()[p]);
        at = tree.tree().parent(p);
    }
    path.reverse();
    path
}

/// Each proposal, in order, follows a live row, carries a token none of that
/// row's followers carries, and is among the likeliest followers a fresh read
/// of its row's whole path ranks — the context before the tree, then the
/// path; then it is made a row, as the requester makes every proposal one.
fn take_flow_proposals(
    model: &mut SplitModel,
    tree: &mut crate::inference::draft_tree::FlowTree,
    context: &[u32],
    proposed: &[(u16, u32)],
) {
    for &(parent, token) in proposed {
        let parent = usize::from(parent);
        assert!(tree.is_live(parent), "a proposal after lost row {parent}");
        assert!(
            !tree.has_child(parent, token),
            "row {parent} carries {token}"
        );
        let mut seq = context.to_vec();
        seq.extend(flow_path(tree, parent));
        let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
        let input = model.tensor_from_ids(&seq).unwrap();
        let out = model
            .forward_verify_all_positions(&input, 0, &kv, "fresh")
            .unwrap();
        let logits = out
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
            .pop()
            .unwrap();
        let mut ranked = logits.clone();
        ranked.sort_by(|a, b| b.total_cmp(a));
        assert!(
            logits[token as usize] >= ranked[super::super::draft::TREE_CHILDREN - 1] - 1e-3,
            "{token} after row {parent} is not among its likeliest followers"
        );
        tree.push(parent, token).unwrap();
    }
}

/// A tree that FLOWS (#134 (b), `SplitModel::grow_flow`), grown over several
/// calls while the requester changes it — proposals made rows, the walk moved
/// down, a draw no row carried added, the tree rebased onto the walk's row —
/// proposes only what a fresh read of each row's path ranks likeliest, only
/// after live rows; one call grows a branch as many rows deep as it has
/// levels; its cache holds exactly the rows it has read; and a cache that no
/// longer holds them is refused.
/// A copied row (`FlowOp::Copy`) is a row as a push is, but rated by the
/// drafter's own guess once its parent is read. Copied with a token the
/// drafter does not list after the root, its followers come after every one of
/// the root's own; a push — as sure as its parent — would put its followers
/// among them.
#[test]
fn a_copied_row_is_rated_by_the_drafters_own_guess() {
    use super::super::draft::TREE_CHILDREN;
    use crate::inference::draft_tree::FlowOp;
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let context = prompt[..7].to_vec();
    let grow_flow =
        |model: &mut SplitModel, session: &mut FlowGrowth, ops: &[FlowOp], rows: usize| {
            model
                .grow_flow(
                    &kv,
                    "c",
                    session,
                    FlowStep {
                        ops,
                        rows,
                        levels: 1,
                        live_most: 128,
                    },
                    &greedy(),
                    None,
                    0,
                )
                .unwrap()
        };
    // The root's own followers, from a session of their own.
    let mut probe = model
        .start_flow(&kv, "c-probe", 0, &context, prompt[7], 0)
        .unwrap();
    let listed: Vec<u32> = model
        .grow_flow(
            &kv,
            "c-probe",
            &mut probe,
            FlowStep {
                ops: &[],
                rows: TREE_CHILDREN,
                levels: 1,
                live_most: 128,
            },
            &greedy(),
            None,
            0,
        )
        .unwrap()
        .into_iter()
        .map(|(_, t)| t)
        .collect();
    assert_eq!(listed.len(), TREE_CHILDREN);
    let unlisted = (0..VOCAB as u32).find(|t| !listed.contains(t)).unwrap();

    // The first call reads the root and proposes nothing; then a copied row
    // the drafter does not list, and a call taking as many rows as the root has
    // followers.
    let mut session = model
        .start_flow(&kv, "c", 0, &context, prompt[7], 0)
        .unwrap();
    assert!(grow_flow(&mut model, &mut session, &[], 0).is_empty());
    let grown = grow_flow(
        &mut model,
        &mut session,
        &[FlowOp::Copy {
            parent: 0,
            token: unlisted,
        }],
        TREE_CHILDREN,
    );
    assert_eq!(grown.len(), TREE_CHILDREN);
    assert!(
        grown.iter().all(|&(parent, _)| parent == 0),
        "the copied row's followers came before the root's own: {grown:?}"
    );
    let mut theirs: Vec<u32> = grown.iter().map(|&(_, t)| t).collect();
    let mut listed = listed;
    theirs.sort_unstable();
    listed.sort_unstable();
    assert_eq!(theirs, listed);
}

#[test]
fn a_tree_that_flows_proposes_what_a_fresh_read_of_each_path_ranks_likeliest() {
    use crate::inference::draft_tree::{FlowOp, FlowTree};
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt: Vec<u32> = vec![3, 17, 99, 250, 7, 41, 12, 5];
    let mut context = prompt[..7].to_vec();
    let mut session = model
        .start_flow(&kv, "f", 0, &context, prompt[7], 0)
        .unwrap();
    let mut mine = FlowTree::new(context.len(), prompt[7]).unwrap();
    let grow_flow = |model: &mut SplitModel,
                     session: &mut FlowGrowth,
                     ops: &[FlowOp],
                     rows: usize,
                     levels: usize| {
        model
            .grow_flow(
                &kv,
                "f",
                session,
                FlowStep {
                    ops,
                    rows,
                    levels,
                    live_most: 128,
                },
                &greedy(),
                None,
                0,
            )
            .unwrap()
    };

    // The first call reads the root and proposes its followers, the greedy
    // guess first.
    let first = grow_flow(&mut model, &mut session, &[], 6, 1);
    assert_eq!(first.len(), 6);
    assert!(first.iter().all(|&(p, _)| p == 0));
    assert_eq!(first[0].1, fresh(&mut model, &prompt, 1)[0]);
    take_flow_proposals(&mut model, &mut mine, &context, &first);
    assert_eq!(kv.request_positions(model.kv_model_key(), "f"), 8);

    // Its proposals are rows of both trees; the next call reads them and
    // proposes on.
    let second = grow_flow(&mut model, &mut session, &[], 12, 1);
    assert_eq!(second.len(), 12);
    take_flow_proposals(&mut model, &mut mine, &context, &second);
    assert_eq!(kv.request_positions(model.kv_model_key(), "f"), 7 + 7);

    // The walk takes row 1, whose draw no row carried: that draw is a row of
    // its own and the walk stands on it, so every proposal follows it. Three
    // levels of TREE_CHILDREN: the first takes every follower the draw has,
    // so the second can only follow the first's rows — one call grows the
    // branch more than one row deep.
    let mut ops = vec![FlowOp::Advance { row: 1 }];
    mine.advance(1).unwrap();
    let missed = (0..VOCAB as u32).find(|&t| !mine.has_child(1, t)).unwrap();
    let bonus = mine.push(1, missed).unwrap();
    mine.advance(bonus).unwrap();
    ops.extend([
        FlowOp::Push {
            parent: 1,
            token: missed,
        },
        FlowOp::Advance { row: bonus as u16 },
    ]);
    let children = super::super::draft::TREE_CHILDREN;
    let third = grow_flow(&mut model, &mut session, &ops, children, 3);
    assert_eq!(third.len(), 3 * children);
    let level_one = bonus + 1..bonus + 1 + children;
    assert!(third[..children]
        .iter()
        .all(|&(p, _)| usize::from(p) == bonus));
    assert!(
        third[children..2 * children]
            .iter()
            .all(|&(p, _)| level_one.contains(&usize::from(p))),
        "the second level follows the first's rows"
    );
    take_flow_proposals(&mut model, &mut mine, &context, &third);
    // The last level is read by the next call.
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "f"),
        mine.base() + mine.len() - children
    );

    // Rebased onto a row below the draw: the path before it is context, and
    // the next call proposes after the kept rows as a fresh read of the whole
    // sequence does.
    let next = level_one.start;
    let ops = [FlowOp::Advance { row: next as u16 }, FlowOp::Rebase];
    mine.advance(next).unwrap();
    let rebased = mine.rebase().unwrap();
    context.extend(&rebased.path);
    assert_eq!(rebased.path.len(), 3, "row 0, row 1 and the draw");
    assert_eq!(mine.base(), 10);
    let fourth = grow_flow(&mut model, &mut session, &ops, 8, 1);
    assert_eq!(fourth.len(), 8);
    take_flow_proposals(&mut model, &mut mine, &context, &fourth);
    assert_eq!(
        (session.flow().base(), session.flow().tokens()),
        (mine.base(), mine.tokens())
    );
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "f"),
        mine.base() + mine.len() - fourth.len()
    );

    // A cache that lost the rows is refused, never read from.
    kv.truncate_request_to(model.kv_model_key(), "f", 4)
        .unwrap();
    let refused = model.grow_flow(
        &kv,
        "f",
        &mut session,
        FlowStep {
            ops: &[],
            rows: 4,
            levels: 1,
            live_most: 128,
        },
        &greedy(),
        None,
        0,
    );
    assert!(matches!(
        refused,
        Err(crate::error::SwarmError::ServiceUnavailable(_))
    ));
}

/// A first segment of one layer, with the token table, and a last of one
/// layer, with the output head, of a three-layer model whose middle layer
/// runs on another computer — a chain this node starts and ends.
fn chain_ends() -> (SplitModel, SplitModel) {
    let hidden = 128;
    let mut first = make_test_split_model(1, hidden);
    let table = Tensor::randn(0f32, 1.0, (VOCAB, hidden), &Device::Cpu).unwrap();
    first.tok_embeddings = Some(TokenEmbedding::Dense(candle_nn::Embedding::new(
        table, hidden,
    )));
    first.total_layers = 3;
    first.layer_end = 1;
    first.kv_model_key = "0-1-3".into();
    let mut last = make_test_split_model(1, hidden);
    last.norm = Some(make_rms_norm_dim(hidden, &Device::Cpu));
    let w = (Tensor::randn(0f32, 1.0, (VOCAB, hidden), &Device::Cpu).unwrap() * 0.5).unwrap();
    last.output = Some(
        QMatMul::from_qtensor(
            candle_core::quantized::QTensor::quantize(&w, candle_core::quantized::GgmlDType::F32)
                .unwrap(),
        )
        .unwrap(),
    );
    last.total_layers = 3;
    last.layer_start = 2;
    last.layer_end = 3;
    last.kv_model_key = "2-3-3".into();
    (first, last)
}

/// A guess at the head of such a chain reads exactly what its two ends read
/// a token at a time with the stand-in between them — each guess the greedy
/// pick, read back in for the next — and leaves both ends' caches where the
/// reply is (`guesses_through_far_run`).
#[test]
fn a_guess_through_a_far_run_is_the_two_ends_read_step_by_step() {
    let (mut first, mut last) = chain_ends();
    let bridge = SkipBridge::Shift {
        run_end: 2,
        shift: (Tensor::randn(0f32, 1.0, (128,), &Device::Cpu).unwrap() * 0.3).unwrap(),
    };
    let step = |first: &mut SplitModel,
                last: &mut SplitModel,
                kv: &KvCacheStore,
                ids: &[u32],
                at: usize|
     -> u32 {
        let input = first.tensor_from_ids(ids).unwrap();
        let h = first.forward(&input, at, kv, "r").unwrap();
        let h = bridge.apply(&h).unwrap();
        let logits: Vec<f32> = last
            .forward(&h, at, kv, "r")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        crate::inference::sampling::argmax(&logits)
    };
    let prompt = [3u32, 17, 42, 9, 101];
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let next = step(&mut first, &mut last, &kv, &prompt, 0);
    let got = guesses_through_far_run(
        &mut first,
        &mut last,
        &bridge,
        &kv,
        "r",
        prompt.len(),
        next,
        4,
    )
    .unwrap();
    for key in ["0-1-3", "2-3-3"] {
        assert_eq!(
            kv.request_positions(key, "r"),
            prompt.len(),
            "{key} cut back"
        );
    }
    let fresh = KvCacheStore::new(std::time::Duration::from_secs(60));
    step(&mut first, &mut last, &fresh, &prompt, 0);
    let mut want = Vec::new();
    let mut token = next;
    for j in 0..4 {
        token = step(&mut first, &mut last, &fresh, &[token], prompt.len() + j);
        want.push(token);
    }
    assert_eq!(got, want);
}
