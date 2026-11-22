//! A tree of guesses read in one forward (#134, `inference::draft_tree`): each
//! row must come out exactly as the model reads that row's own path from
//! nothing — the property every segment of a tree check, and the drafter that
//! grows one, rests on.

use super::super::token_embedding::TokenEmbedding;
use super::super::*;
use super::common::*;
#[cfg(feature = "candle-cuda")]
use crate::inference::cuda_graph::StepShape;
use crate::inference::draft_tree::DraftTree;
use candle_core::{Device, Tensor};

const VOCAB: usize = 256;

/// A whole model with random weights on the processor (`drafting.rs`'s).
fn whole_model() -> SplitModel {
    whole_model_on(&Device::Cpu)
}

fn whole_model_on(device: &Device) -> SplitModel {
    let hidden = 128;
    let mut m = make_test_split_model_on(3, hidden, device.clone());
    let table = Tensor::randn(0f32, 1.0, (VOCAB, hidden), device).unwrap();
    m.tok_embeddings = Some(TokenEmbedding::Dense(candle_nn::Embedding::new(
        table, hidden,
    )));
    m.norm = Some(make_rms_norm_dim(hidden, device));
    let w = (Tensor::randn(0f32, 1.0, (VOCAB, hidden), device).unwrap() * 0.5).unwrap();
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

/// The largest gap, NaN if any gap is: `f32::max` drops a NaN, so a kernel
/// answering NaN would pass a tolerance folded with it (gotcha #882).
fn nan_max(worst: f32, gap: f32) -> f32 {
    if worst.is_nan() || gap.is_nan() {
        f32::NAN
    } else {
        worst.max(gap)
    }
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    let worst = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, nan_max);
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
        .with_draft_tree(t, None, |m| {
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
        .with_draft_tree(t, None, |m| {
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

/// A level of a drafter's tree captured as a CUDA graph (#134) reads what the
/// same level read op by op does — and WAS captured, the decode graph's own
/// count says so: a conversation's first step and the first forward of a
/// shape never are, so only the tree's later levels can be. Four levels of 8
/// rows (each row a child of a row of the level before), the model held on the
/// card, run with capture off and then on.
///
/// ⚠ Gated: run it on a card with `--nocapture` and look for the SKIPPED line
/// before believing a pass — and with `--test-threads=1` beside other card
/// tests: another test's card calls during a capture refuse it (seen once in
/// a parallel run; alone it passed 3 of 3).
#[cfg(feature = "candle-cuda")]
#[test]
fn a_captured_tree_level_reads_as_the_ordinary_one() {
    use crate::inference::cuda_graph::DecodeGraphSlot;
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED captured tree level: no CUDA device ({e})");
            return;
        }
    };
    let mut model = whole_model_on(&card);
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    const LEVEL: usize = crate::inference::cuda_graph::MAX_POSITIONS;
    let run = |model: &mut SplitModel, rid: &str| -> Vec<Vec<f32>> {
        let prompt = model.tensor_from_ids(&PROMPT).unwrap();
        model.forward(&prompt, 0, &kv, rid).unwrap();
        let step = model.tensor_from_ids(&[42]).unwrap();
        model.forward(&step, PROMPT.len(), &kv, rid).unwrap();
        let base = PROMPT.len() + 1;
        let mut parents: Vec<Option<u16>> = Vec::new();
        let mut rows = Vec::new();
        for level in 0..4 {
            let before = parents.len();
            for i in 0..LEVEL {
                parents.push((level > 0).then(|| (before - LEVEL + i) as u16));
            }
            let tokens: Vec<u32> = (0..LEVEL)
                .map(|i| (level * 31 + i * 7 + 1) as u32)
                .collect();
            let tree = DraftTree::new(base, parents.clone()).unwrap();
            let input = model.tensor_from_ids(&tokens).unwrap();
            let out = model
                .with_draft_tree(tree, None, |m| {
                    m.forward_verify_all_positions(&input, base + before, &kv, rid)
                })
                .unwrap()
                .unwrap();
            rows.extend(rows_of(&out.to_device(&Device::Cpu).unwrap()));
        }
        rows
    };
    model.decode_graph = DecodeGraphSlot::Off;
    let ordinary = run(&mut model, "op-by-op");
    model.decode_graph = DecodeGraphSlot::Unchecked;
    // Its graphs made as the idle recording makes them: a reply leaves a
    // shape's first graph to it (#357 (d)).
    let _ = model.widths_to_record();
    model.set_recording_widths(true);
    let captured = run(&mut model, "captured");
    model.set_recording_widths(false);
    let DecodeGraphSlot::On(graph) = &model.decode_graph else {
        panic!("the card model's decode steps were not to be captured");
    };
    let (launched, refused) = graph.launches();
    assert_eq!(refused, 0, "a captured tree level was refused");
    assert!(
        launched >= 2,
        "the tree's later levels were not captured ({launched} launched)"
    );
    assert_eq!(ordinary.len(), captured.len());
    for (i, (a, b)) in ordinary.iter().zip(&captured).enumerate() {
        let worst = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, nan_max);
        assert!(worst < 1e-3, "row {i}: captured differs by {worst}");
    }
}

/// A split's first segment on the card: the same three layers as
/// [`whole_model_on`], of a six-layer model, so no norm or head — its forwards
/// carry token ids and send rows on.
#[cfg(feature = "candle-cuda")]
fn first_segment_on(device: &Device) -> SplitModel {
    let mut m = whole_model_on(device);
    m.norm = None;
    m.output = None;
    m.total_layers = 6;
    m.kv_model_key = "0-3-6".into();
    m
}

/// One forward of `input` at `at` as a reply makes it: a model that ends the
/// reply as the worker's generate and the drafter do, a first segment as the
/// worker answers a peer's forward.
#[cfg(feature = "candle-cuda")]
fn reply_forward(model: &mut SplitModel, kv: &KvCacheStore, input: &Tensor, at: usize, rows: bool) {
    if !model.is_last() {
        model
            .forward_prompt_in_chunks_keeping(input, at, kv, "reply", None, false, 0, None)
            .unwrap();
    } else if rows {
        model
            .forward_verify_all_positions(input, at, kv, "reply")
            .unwrap();
    } else {
        model.forward(input, at, kv, "reply").unwrap();
    }
}

/// Record what the idle worker records (`model_worker::record_check_widths`),
/// then run a reply through every shape it named and count the forwards that
/// made a graph. Each chain width runs as a reply's check does, cut back to its
/// first position; each tree row count as a drafter's later level does — read
/// after a level already in the cache, then cut back.
#[cfg(feature = "candle-cuda")]
fn one_offs_in_a_reply_after_the_recording(
    model: &mut SplitModel,
    widest: usize,
) -> (Vec<StepShape>, u64) {
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let widths = model.widths_to_record();
    let finished =
        crate::inference::model_worker::record_check_widths(model, &kv, &widths, &|| true).unwrap();
    assert!(finished, "nothing asked the recording to stop");
    for &w in &widths {
        assert!(
            model.has_step_graph(w),
            "{w} has no graph after the recording"
        );
    }
    assert_eq!(
        kv.active_entries(),
        0,
        "the recording left its cache behind"
    );
    // A reply: its prompt, its first step (a one-off in every reply: nothing
    // of its own ran before it), then each shape as a reply runs it.
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "reply").unwrap();
    let step = model.tensor_from_ids(&[42]).unwrap();
    reply_forward(model, &kv, &step, PROMPT.len(), false);
    let before = model.one_off_forwards();
    let mut at = PROMPT.len() + 1;
    // Every shape a reply may ask for — never only the ones recorded, or a
    // shape the recording left out would never be run here and its one-off
    // never counted.
    let reply_shapes = crate::inference::cuda_graph::shapes_to_record(widest, true, |_| false);
    for &w in reply_shapes.iter().rev() {
        let ids: Vec<u32> = (0..w.positions as u32).map(|i| 7 + i).collect();
        let input = model.tensor_from_ids(&ids).unwrap();
        if w.tree {
            // A level of two rows already read, then this one's rows: row 0
            // follows the context, row 1 row 0, the new rows rows 0 and 1 in turn.
            let lead = model.tensor_from_ids(&[5, 9]).unwrap();
            let head = DraftTree::new(at, vec![None, Some(0)]).unwrap();
            model
                .with_draft_tree(head, None, |m| reply_forward(m, &kv, &lead, at, true))
                .unwrap();
            let mut parents = vec![None, Some(0)];
            parents.extend((0..w.positions).map(|i| Some((i % 2) as u16)));
            let tree = DraftTree::new(at, parents).unwrap();
            model
                .with_draft_tree(tree, None, |m| reply_forward(m, &kv, &input, at + 2, true))
                .unwrap();
            kv.truncate_request_to(model.kv_model_key(), "reply", at)
                .unwrap();
        } else {
            reply_forward(model, &kv, &input, at, w.positions > 1);
            if w.positions > 1 {
                kv.truncate_request_to(model.kv_model_key(), "reply", at + 1)
                    .unwrap();
            }
            at += 1;
        }
    }
    (widths, model.one_off_forwards() - before)
}

/// A reply's checks and tree levels find their graphs already made once an
/// idle worker has recorded them (`model_worker::record_check_widths`): no
/// forward of any shape a reply asks for is then a one-off. Without the
/// recording each shape's first forward runs the ordinary way and the next
/// makes its graph — 0.3-8 s each under WSL2, inside the reply (#428, card runs
/// 8-9) — and a tree's row counts were not recorded at all (#357 (b): 289-958
/// ms each inside the 14B split's replies, 2026-11-14). The recording's own
/// request leaves no cache behind.
///
/// ⚠ Gated: run it on a card with `--nocapture` and look for the SKIPPED line
/// before believing a pass, with `--test-threads=1` beside other card tests.
#[cfg(feature = "candle-cuda")]
#[test]
fn a_reply_after_the_idle_recording_makes_no_graph() {
    use crate::inference::cuda_graph::MAX_POSITIONS;
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED idle recording: no CUDA device ({e})");
            return;
        }
    };
    let mut model = whole_model_on(&card);
    // Several positions are captured only where flash attention reads no mask.
    let widest = if cfg!(feature = "flash-attn") {
        MAX_POSITIONS
    } else {
        1
    };
    let (widths, one_offs) = one_offs_in_a_reply_after_the_recording(&mut model, widest);
    assert_eq!(
        one_offs, 0,
        "a forward after the recording made its graph inside the reply: {widths:?}"
    );
    assert_eq!(
        widths.first(),
        Some(&StepShape::chain(widest)),
        "a whole model on the card names every width, widest first: {widths:?}"
    );
    assert!(
        (1..=MAX_POSITIONS).all(|r| widths.contains(&StepShape::tree(r))),
        "a whole model whose trees are captured names every tree row count: {widths:?}"
    );
}

/// A reply BEFORE the idle recording makes no graph (#357 (d)): a worker asked
/// back to back, or for its first reply after a load, is never idle, and each
/// graph its replies made cost up to 9.8 s under WSL2 — the 7B's v0.3.299
/// card A/B, a third of its replies 3-8x slow. A shape with no graph
/// runs the ordinary way and is left to the recording; once that has run, the
/// reply's next steps update its graph in place. Red under
/// `SWARMLLM_CUDA_GRAPH_MADE=in-replies` (the step's second forward makes it).
///
/// ⚠ Gated as above.
#[cfg(feature = "candle-cuda")]
#[test]
fn a_reply_before_the_idle_recording_makes_no_graph() {
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED reply before the recording: no CUDA device ({e})");
            return;
        }
    };
    let mut model = whole_model_on(&card);
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "reply").unwrap();
    let mut at = PROMPT.len();
    // The first step learns the output's shape; before #357 (d) the second
    // made the graph, inside the reply.
    for id in 0..4u32 {
        let step = model.tensor_from_ids(&[40 + id]).unwrap();
        reply_forward(&mut model, &kv, &step, at, false);
        at += 1;
    }
    let step = StepShape::chain(1);
    assert!(
        !model.has_step_graph(step),
        "a reply made the step's graph before the idle recording"
    );
    let widths = model.widths_to_record();
    assert!(
        widths.contains(&step),
        "the recording names the step: {widths:?}"
    );
    let finished =
        crate::inference::model_worker::record_check_widths(&mut model, &kv, &widths, &|| true)
            .unwrap();
    assert!(finished, "nothing asked the recording to stop");
    assert!(model.has_step_graph(step), "the recording made no graph");
    let before = model.one_off_forwards();
    for id in 0..3u32 {
        let next = model.tensor_from_ids(&[50 + id]).unwrap();
        reply_forward(&mut model, &kv, &next, at, false);
        at += 1;
    }
    assert_eq!(
        model.one_off_forwards() - before,
        0,
        "the reply's steps after the recording did not update its graph"
    );
}

/// The same for a split's FIRST segment (#357 (b)): its forwards carry token
/// ids, so the idle worker records its checks and tree levels too — before it,
/// a first segment named nothing and every flowing tree's row count made its
/// graph inside a reply.
///
/// ⚠ Gated as above.
#[cfg(feature = "candle-cuda")]
#[test]
fn a_first_segment_records_its_shapes_before_a_reply_needs_them() {
    use crate::inference::cuda_graph::MAX_POSITIONS;
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED first segment recording: no CUDA device ({e})");
            return;
        }
    };
    let mut model = first_segment_on(&card);
    assert!(model.is_first() && !model.is_last());
    let widest = if cfg!(feature = "flash-attn") {
        MAX_POSITIONS
    } else {
        1
    };
    let (widths, one_offs) = one_offs_in_a_reply_after_the_recording(&mut model, widest);
    assert_eq!(
        one_offs, 0,
        "a first segment's forward after the recording made its graph inside the reply: {widths:?}"
    );
    assert!(
        widths.contains(&StepShape::chain(1))
            && (1..=MAX_POSITIONS).all(|r| widths.contains(&StepShape::tree(r))),
        "a first segment names its step and every tree row count: {widths:?}"
    );
}

/// A tree WIDER than the per-row kernel's rows, read on the card in one
/// forward — a worker's check of a flow's levels — comes out row by row as
/// each row's own path read fresh, and was read by the TILED kernel
/// (`decode_attn::tiled_tree_calls`; `SWARMLLM_TREE_ATTN_TILED=0` turns this
/// test red). 24 rows, branching (every fifth row starts afresh), over a half
/// cache.
///
/// ⚠ Gated: run it on a card with `--nocapture` and look for the SKIPPED line
/// before believing a pass.
#[cfg(feature = "candle-cuda")]
#[test]
fn a_wide_tree_on_the_card_reads_what_each_path_reads() {
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED wide tree on the card: no CUDA device ({e})");
            return;
        }
    };
    let mut model = whole_model_on(&card);
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let prompt = model.tensor_from_ids(&PROMPT).unwrap();
    model.forward(&prompt, 0, &kv, "w").unwrap();
    const ROWS: usize = 24;
    let parents: Vec<Option<u16>> = (0..ROWS)
        .map(|i| (i % 5 != 0).then(|| ((i * 7 + 3) % i) as u16))
        .collect();
    let tokens: Vec<u32> = (0..ROWS)
        .map(|i| (i * 37 + 11) as u32 % VOCAB as u32)
        .collect();
    let input = model.tensor_from_ids(&tokens).unwrap();
    let t = DraftTree::new(PROMPT.len(), parents.clone()).unwrap();
    let before = crate::inference::decode_attn::tiled_tree_calls();
    let out = model
        .with_draft_tree(t, None, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "w")
        })
        .unwrap()
        .unwrap();
    let tiled = crate::inference::decode_attn::tiled_tree_calls() - before;
    assert!(
        tiled >= 3,
        "the tiled tree kernel read {tiled} of the 3 layers' attention"
    );
    let rows = rows_of(&out.to_device(&Device::Cpu).unwrap());
    assert_eq!(rows.len(), ROWS);
    for (row, read) in rows.iter().enumerate() {
        let mut seq = PROMPT.to_vec();
        seq.extend(path_of(&tokens, &parents, row));
        let fresh = read_fresh(&mut model, &seq);
        let worst = read
            .iter()
            .zip(&fresh)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, nan_max);
        assert!(
            worst < 1e-3,
            "row {row}: differs from its path read fresh by {worst}"
        );
    }
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
        .with_draft_tree(first, None, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "g")
        })
        .unwrap()
        .unwrap();
    let whole = DraftTree::new(PROMPT.len(), parents.clone()).unwrap();
    let input = model.tensor_from_ids(&tokens[3..]).unwrap();
    let out = model
        .with_draft_tree(whole, None, |m| {
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
            .with_draft_tree(shape, None, |m| {
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
            None,
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
        .with_draft_tree(t, None, |m| {
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
    assert!(model.with_draft_tree(t, None, |_| ran = true).is_err());
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
        .with_draft_tree(t, None, |m| {
            m.forward_verify_all_positions(&input, PROMPT.len(), &kv, "r")
        })
        .unwrap();
    assert!(out.is_err());
    assert_eq!(
        kv.request_positions(model.kv_model_key(), "r"),
        PROMPT.len()
    );
}

/// A recording stopped by a message just after its LAST shape's graph was
/// made is finished, not stopped: the next idle moment finds nothing left to
/// record and never asks again, so a recording reported stopped there was
/// never marked done, and every shape left to it — a drafter's tree levels —
/// ran uncaptured for the worker's life (38-39 ms a forward against 27-28,
/// Qwen3.5-0.8B on the TH↔AU link, 2026-11-23). One shape: the recording's
/// first forward of it runs the ordinary way and its second makes the graph;
/// a message is waiting from then on.
///
/// ⚠ Gated as above.
#[cfg(feature = "candle-cuda")]
#[test]
fn a_recording_whose_last_graph_is_made_is_finished_though_a_message_came() {
    let card = match Device::new_cuda(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SKIPPED recording stopped after its last graph: no CUDA device ({e})");
            return;
        }
    };
    let mut model = whole_model_on(&card);
    let kv = KvCacheStore::new(std::time::Duration::from_secs(60));
    let step = StepShape::chain(1);
    assert!(
        model.widths_to_record().contains(&step),
        "the recording names the step"
    );
    let asked = std::cell::Cell::new(0);
    let keep_going = || {
        asked.set(asked.get() + 1);
        asked.get() <= 2
    };
    let finished =
        crate::inference::model_worker::record_check_widths(&mut model, &kv, &[step], &keep_going)
            .unwrap();
    assert!(model.has_step_graph(step), "the recording made no graph");
    assert!(
        finished,
        "the recording made its last graph and was still reported stopped"
    );
}
