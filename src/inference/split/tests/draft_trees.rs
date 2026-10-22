//! A tree of guesses read in one forward (#134, `inference::draft_tree`): each
//! row must come out exactly as the model reads that row's own path from
//! nothing — the property every segment of a tree check, and the drafter that
//! grows one, rests on.

use super::super::token_embedding::TokenEmbedding;
use super::super::*;
use super::common::*;
use crate::inference::draft_tree::DraftTree;
use candle_core::{Device, Tensor};

const VOCAB: usize = 256;

/// A whole model with random weights on the processor (`drafting.rs`'s).
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

fn rows_of(logits: &Tensor) -> Vec<Vec<f32>> {
    logits.squeeze(0).unwrap().to_vec2::<f32>().unwrap()
}

/// The last position's logits for `seq` read from an empty cache.
fn read_fresh(model: &mut SplitModel, seq: &[u32]) -> Vec<f32> {
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let input = model.tensor_from_ids(seq).unwrap();
    let out = model
        .forward_verify_all_positions(&input, 0, &kv, "fresh")
        .unwrap();
    rows_of(&out).pop().unwrap()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(worst < 1e-4, "{what}: max abs diff {worst}");
}

const PROMPT: [u32; 5] = [3, 17, 99, 250, 7];

/// Row 0 is the check's own token; rows 1 and 2 follow it; 3 follows 1; 4
/// follows 2; 5 follows 3.
fn tree() -> (Vec<u32>, Vec<Option<u16>>) {
    (
        vec![5, 9, 11, 13, 15, 21],
        vec![None, Some(0), Some(0), Some(1), Some(2), Some(3)],
    )
}

fn path_of(tokens: &[u32], parents: &[Option<u16>], row: usize) -> Vec<u32> {
    let mut path = vec![tokens[row]];
    let mut at = parents[row];
    while let Some(p) = at {
        path.push(tokens[p as usize]);
        at = parents[p as usize];
    }
    path.reverse();
    path
}

#[test]
fn every_row_of_a_tree_reads_what_its_own_path_reads() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "t").unwrap();

    let (tokens, parents) = tree();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let t = DraftTree::new(PROMPT.len(), parents.clone()).unwrap();
    let out = model
        .with_draft_tree(t, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "t")
        })
        .unwrap()
        .unwrap();
    let rows = rows_of(&out);
    assert_eq!(rows.len(), tokens.len());
    for (row, read) in rows.iter().enumerate() {
        let mut seq = PROMPT.to_vec();
        seq.extend(path_of(&tokens, &parents, row));
        assert_close(read, &read_fresh(&mut model, &seq), &format!("row {row}"));
    }
    // The control: the same rows read as a chain see their siblings, and the
    // rows that have one come out different — the mask is what made them agree.
    let chain = model
        .forward_verify_all_positions(&input, PROMPT.len(), &kv, "chain")
        .map(|o| rows_of(&o));
    if let Ok(chain) = chain {
        let differs = (0..tokens.len()).any(|r| {
            chain[r]
                .iter()
                .zip(&rows[r])
                .any(|(a, b)| (a - b).abs() > 1e-3)
        });
        assert!(
            differs,
            "a chain over the same rows read the same — the check proves nothing"
        );
    }
}

/// After a check keeps a path that is not a prefix of the rows, the cache is
/// cut to the part that is, and the rest is read again as a chain: the next
/// position's logits must be the path's own.
#[test]
fn a_kept_path_read_again_continues_as_if_read_alone() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "k").unwrap();
    let (tokens, parents) = tree();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let t = DraftTree::new(PROMPT.len(), parents).unwrap();
    model
        .with_draft_tree(t, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "k")
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "k"),
        PROMPT.len() + 6
    );
    // Kept: row 0 (the check's token, always), then rows 2 and 4 — 11, 15 —
    // which are not rows 1 and 2. Cut to row 0, read 11 and 15 again.
    kv.truncate_request_to(model.kv_model_key(), "k", PROMPT.len() + 1)
        .unwrap();
    let again = model.tensor_from_ids(&[11, 15]).unwrap();
    let out = model
        .forward_verify_all_positions(&again, PROMPT.len() + 1, &kv, "k")
        .unwrap();
    let last = rows_of(&out).pop().unwrap();
    let mut seq = PROMPT.to_vec();
    seq.extend([5, 11, 15]);
    assert_close(&last, &read_fresh(&mut model, &seq), "after the re-read");
}

/// A tree grown over several forwards — the drafter's way — reads its cached
/// rows as keys: the second forward's rows come out as their paths read.
#[test]
fn a_tree_grown_a_step_at_a_time_reads_its_earlier_rows() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "g").unwrap();
    let (tokens, parents) = tree();
    let first = DraftTree::new(PROMPT.len(), parents[..3].to_vec()).unwrap();
    let input = model.tensor_from_ids(&tokens[..3]).unwrap();
    model
        .with_draft_tree(first, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "g")
        })
        .unwrap()
        .unwrap();
    let whole = DraftTree::new(PROMPT.len(), parents.clone()).unwrap();
    let input = model.tensor_from_ids(&tokens[3..]).unwrap();
    let out = model
        .with_draft_tree(whole, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len() + 3, &kv, "g")
        })
        .unwrap()
        .unwrap();
    let rows = rows_of(&out);
    for (i, row) in (3..tokens.len()).enumerate() {
        let mut seq = PROMPT.to_vec();
        seq.extend(path_of(&tokens, &parents, row));
        assert_close(
            &rows[i],
            &read_fresh(&mut model, &seq),
            &format!("row {row}"),
        );
    }
}

/// A tree that FLOWS (#134 (b)): sent a few rows a forward while it grows,
/// with a later row hanging off an EARLY row after other rows have grown
/// elsewhere — the row a requester adds where the model's draw missed every
/// guess. Every row must read as its own path does from nothing, and as the
/// same rows read in ONE forward: the rows in between are in the cache and
/// read by none of its descendants.
#[test]
fn a_tree_read_in_three_forwards_reads_as_one() {
    let mut model = whole_model();
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    // Rows: 0 the check's token; 1, 2 follow 0; 3 follows 1; 4 follows 3;
    // then 5 follows 2 (rows 1, 3, 4 lost) and 6 follows 5.
    let tokens = [5u32, 9, 11, 13, 15, 21, 30];
    let parents = vec![None, Some(0), Some(0), Some(1), Some(3), Some(2), Some(5)];
    let forwards = [0..3, 3..5, 5..7];

    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    model.forward(&prompt, 0, &kv, "f").unwrap();
    let mut flowed = Vec::new();
    for rows in forwards {
        let shape = DraftTree::new(PROMPT.len(), parents[..rows.end].to_vec()).unwrap();
        let input = model.tensor_from_ids(&tokens[rows.clone()]).unwrap();
        let out = model
            .with_draft_tree(shape, |m| {
                m.forward_verify_all_positions(&input, PROMPT.len() + rows.start, &kv, "f")
            })
            .unwrap()
            .unwrap();
        flowed.extend(rows_of(&out));
    }

    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    model.forward(&prompt, 0, &kv, "o").unwrap();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let once = model
        .with_draft_tree(
            DraftTree::new(PROMPT.len(), parents.clone()).unwrap(),
            |m| m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "o"),
        )
        .unwrap()
        .map(|o| rows_of(&o))
        .unwrap();

    assert_eq!(flowed.len(), tokens.len());
    for row in 0..tokens.len() {
        let mut seq = PROMPT.to_vec();
        seq.extend(path_of(&tokens, &parents, row));
        let fresh = read_fresh(&mut model, &seq);
        assert_close(&flowed[row], &fresh, &format!("row {row}, flowed"));
        assert_close(&flowed[row], &once[row], &format!("row {row}, read once"));
    }
    // The control: the late rows read as a chain after the cached ones see
    // the lost branch, and come out different — the shape is what kept them
    // from it.
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    model.forward(&prompt, 0, &kv, "c").unwrap();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let chain = rows_of(
        &model
            .forward_verify_all_positions(&input, PROMPT.len(), &kv, "c")
            .unwrap(),
    );
    assert!(
        (5..7).any(|r| chain[r]
            .iter()
            .zip(&flowed[r])
            .any(|(a, b)| (a - b).abs() > 1e-3)),
        "the late rows read the same as a chain — the check proves nothing"
    );
}

/// A segment configured to read prompts in pieces smaller than a tree reads
/// the tree whole: its mask covers every row, and a piece of it would be
/// refused — on a middle segment that failed every honest tree (security
/// review of #134).
#[test]
fn a_tree_wider_than_the_prompt_chunk_is_read_whole() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "w").unwrap();
    let (tokens, parents) = tree();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let t = DraftTree::new(PROMPT.len(), parents.clone()).unwrap();
    let out = model
        .with_draft_tree(t, |m| {
            m.forward_prompt_in_chunks(&input, PROMPT.len(), &kv, "w", None, false, 2)
        })
        .unwrap()
        .expect("a tree is one forward, whatever the prompt chunk");
    let last = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut seq = PROMPT.to_vec();
    seq.extend(path_of(&tokens, &parents, tokens.len() - 1));
    let fresh = read_fresh(&mut model, &seq);
    assert_close(&last[last.len() - fresh.len()..], &fresh, "the last row");
}

#[test]
fn a_model_whose_layers_cannot_read_a_tree_refuses_it_before_running() {
    let mut model = whole_model();
    if let LayerVariant::Dense(lw) = &mut model.layers[1] {
        lw.sliding_window = Some(4);
    }
    let t = DraftTree::new(3, vec![None, Some(0)]).unwrap();
    let mut ran = false;
    assert!(model.with_draft_tree(t, |_| ran = true).is_err());
    assert!(!ran);
}

/// A tree whose rows do not end where the cache does is refused, and the
/// conversation's cache is left as it was.
#[test]
fn a_tree_that_does_not_fit_the_cache_is_refused_and_the_cache_kept() {
    let mut model = whole_model();
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "r").unwrap();
    // Says it starts at 4 while the cache holds 5.
    let t = DraftTree::new(4, vec![None, Some(0)]).unwrap();
    let input = model.tensor_from_ids(&[5, 9]).unwrap();
    let out = model
        .with_draft_tree(t, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "r")
        })
        .unwrap();
    assert!(out.is_err());
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "r"),
        PROMPT.len()
    );
}
