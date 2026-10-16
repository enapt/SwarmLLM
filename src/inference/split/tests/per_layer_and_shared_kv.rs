//! Gemma 4 E2B/E4B's two mechanisms (#115 (3c)): layers that attend an
//! earlier layer's cache instead of keeping their own (`LayerWeights::kv_source`),
//! and every layer's embedding gated by a per-layer input built from the tokens
//! (`split::per_layer`). Judged against llama.cpp on a tiny file and on the real
//! E2B (`examples/make_tiny_gemma_gguf.py` `PLE=` / `SHARED=`); these pin the
//! mechanism, each failing with it removed.

use super::super::model::SplitModel;
use super::super::token_embedding::TokenEmbedding;
use super::super::*;
use super::common::*;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};

fn store() -> KvCacheStore {
    KvCacheStore::new(std::time::Duration::from_secs(600))
}

fn dense(model: &mut SplitModel, layer: usize) -> &mut LayerWeights {
    match &mut model.layers[layer] {
        LayerVariant::Dense(lw) => lw,
        _ => unreachable!("the test model is dense"),
    }
}

fn values(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn largest_gap(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// A shared layer attends the cache its source layer wrote earlier in the
/// same forward, keeps none of its own, and never reads its own K and V
/// weights — which the real files still carry.
#[test]
fn a_shared_kv_layer_attends_its_source_layers_cache_and_never_its_own() {
    let mut model = make_test_split_model(3, 64);
    dense(&mut model, 2).kv_source = Some(1);
    let key = model.kv_model_key.clone();

    let store = store();
    // The same inputs for every arm: a prompt pass and two decode steps.
    let prompt = Tensor::randn(0f32, 1.0, (1, 5, 64), &Device::Cpu).unwrap();
    let steps: Vec<Tensor> = (0..2)
        .map(|_| Tensor::randn(0f32, 1.0, (1, 1, 64), &Device::Cpu).unwrap())
        .collect();
    let run = |model: &mut SplitModel, id: &str| -> Vec<f32> {
        let mut out = values(&model.forward(&prompt, 0, &store, id).unwrap());
        for (i, step) in steps.iter().enumerate() {
            out.extend(values(&model.forward(step, 5 + i, &store, id).unwrap()));
        }
        out
    };

    let shared = run(&mut model, "shared");
    {
        let entry = store.get_or_create(&key, "shared", 3);
        assert!(entry.layers[2].is_none(), "a shared layer keeps no cache");
        let source = entry.layers[1].as_ref().expect("its source keeps one");
        assert_eq!(source.current_seq_len(), 7, "written once per position");
    }

    // New K and V weights on the shared layer change nothing: they are not read.
    dense(&mut model, 2).attention_wk = make_qmatmul(64, 64, &Device::Cpu);
    dense(&mut model, 2).attention_wv = make_qmatmul(64, 64, &Device::Cpu);
    let reweighted = run(&mut model, "reweighted");
    assert_eq!(shared, reweighted, "the shared layer read its own K or V");

    // The same layer keeping its own cache answers differently — the source's
    // keys are what made the first answer.
    dense(&mut model, 2).kv_source = None;
    let own = run(&mut model, "own");
    assert!(
        largest_gap(&shared, &own) > 1e-3,
        "a layer on its own cache answered as the shared one"
    );
}

/// The batched decode step hands a shared layer each request's source cache,
/// as the single-request path does.
#[test]
fn a_batched_decode_step_reads_each_requests_source_cache() {
    let mut model = make_test_split_model(3, 64);
    dense(&mut model, 2).kv_source = Some(1);
    let store = store();
    let prompts: Vec<Tensor> = (0..2)
        .map(|_| Tensor::randn(0f32, 1.0, (1, 4, 64), &Device::Cpu).unwrap())
        .collect();
    let steps: Vec<Tensor> = (0..2)
        .map(|_| Tensor::randn(0f32, 1.0, (1, 1, 64), &Device::Cpu).unwrap())
        .collect();
    for (i, prompt) in prompts.iter().enumerate() {
        model
            .forward(prompt, 0, &store, &format!("seq{i}"))
            .unwrap();
        model
            .forward(prompt, 0, &store, &format!("bat{i}"))
            .unwrap();
    }
    let one_at_a_time: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            values(
                &model
                    .forward(&steps[i], 4, &store, &format!("seq{i}"))
                    .unwrap(),
            )
        })
        .collect();
    let ids = ["bat0".to_string(), "bat1".to_string()];
    let items: Vec<BatchItem<'_>> = (0..2)
        .map(|i| BatchItem {
            input: &steps[i],
            index_pos: 4,
            request_id: &ids[i],
        })
        .collect();
    let batched = model.forward_batch(&items, &store).unwrap();
    for i in 0..2 {
        let gap = largest_gap(&one_at_a_time[i], &values(&batched[i]));
        assert!(gap < 1e-4, "request {i}: batched differs by {gap}");
    }
    let entry = store.get_or_create(&model.kv_model_key.clone(), "bat1", 3);
    assert!(entry.layers[2].is_none());
    assert_eq!(entry.layers[1].as_ref().unwrap().current_seq_len(), 5);
}

/// A first segment of `layers` dense layers with a 16-token vocabulary and,
/// when `per_layer` is set, a per-layer embedding `width` wide on each layer
/// fed by `table` / `model_proj`.
fn first_segment(layers: usize, per_layer: Option<(Tensor, Tensor)>, width: usize) -> SplitModel {
    let hidden = 64;
    let mut model = make_test_split_model(layers, hidden);
    let table = Tensor::randn(0f32, 1.0, (16, hidden), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    model.tok_embeddings = Some(TokenEmbedding::Dense(candle_nn::Embedding::new(
        table, hidden,
    )));
    if let Some((rows, proj)) = per_layer {
        let q = |t: &Tensor| QTensor::quantize(t, GgmlDType::F32).unwrap();
        model.per_layer_inputs = Some(
            PerLayerInputs::new(
                q(&rows),
                QMatMul::from_qtensor(q(&proj)).unwrap(),
                make_rms_norm_dim(width, &Device::Cpu),
                layers,
                width,
            )
            .unwrap(),
        );
        for layer in 0..layers {
            dense(&mut model, layer).per_layer_embedding = Some(Box::new(PerLayerEmbedding {
                inp_gate: make_qmatmul(hidden, width, &Device::Cpu),
                proj: make_qmatmul(width, hidden, &Device::Cpu),
                post_norm: make_rms_norm_dim(hidden, &Device::Cpu),
            }));
        }
    }
    model
}

/// Each layer's embedding is added as its per-layer input gates it: with an
/// input of zero it adds nothing — the model without one, bit for bit — and
/// with a real input the answer moves. A forward that skipped the embedding,
/// or handed a layer no input, could not pass both.
#[test]
fn a_layers_per_layer_input_gates_its_embedding() {
    let (layers, width) = (2, 8);
    let ids = Tensor::new(&[[1u32, 5, 9]], &Device::Cpu).unwrap();
    let run =
        |model: &mut SplitModel, id: &str| values(&model.forward(&ids, 0, &store(), id).unwrap());

    // One model for every arm — its token table and layer weights are random
    // — with only the per-layer parts swapped.
    let mut model = first_segment(
        layers,
        Some((
            Tensor::zeros((16, layers * width), DType::F32, &Device::Cpu).unwrap(),
            Tensor::zeros((layers * width, 64), DType::F32, &Device::Cpu).unwrap(),
        )),
        width,
    );
    let zero_input = run(&mut model, "zero");

    let embeddings: Vec<_> = (0..layers)
        .map(|l| dense(&mut model, l).per_layer_embedding.take())
        .collect();
    model.per_layer_inputs = None;
    let without = run(&mut model, "without");
    assert_eq!(zero_input, without, "a zero input added something");

    for (l, e) in embeddings.into_iter().enumerate() {
        dense(&mut model, l).per_layer_embedding = e;
    }
    model.per_layer_inputs = first_segment(
        layers,
        Some((
            Tensor::randn(0f32, 1.0, (16, layers * width), &Device::Cpu).unwrap(),
            Tensor::randn(0f32, 0.3, (layers * width, 64), &Device::Cpu).unwrap(),
        )),
        width,
    )
    .per_layer_inputs;
    let with = run(&mut model, "with");
    assert!(
        largest_gap(&with, &without) > 1e-3,
        "the per-layer input changed nothing"
    );
}

/// Hidden states handed to a model whose layers read per-layer inputs — a
/// pre-embedded forward, or a later segment's — are refused: the tokens the
/// inputs are built from are not there.
#[test]
fn hidden_states_without_their_tokens_are_refused() {
    let width = 8;
    let mut model = first_segment(
        2,
        Some((
            Tensor::randn(0f32, 1.0, (16, 2 * width), &Device::Cpu).unwrap(),
            Tensor::randn(0f32, 0.3, (2 * width, 64), &Device::Cpu).unwrap(),
        )),
        width,
    );
    let hidden = Tensor::randn(0f32, 1.0, (1, 3, 64), &Device::Cpu).unwrap();
    let err = model
        .forward_pre_embedded(&hidden, 0, &store(), "pre")
        .expect_err("ran every layer without its input");
    assert!(err.to_string().contains("per-layer inputs"), "{err}");
}

/// The inputs as llama.cpp builds them (`build_inp_per_layer`,
/// `project_per_layer_inputs`), worked by hand for one token: the table's row
/// × √width, plus the embedding's projection ÷ √n_embd normed per layer, the
/// sum ÷ √2.
#[test]
fn per_layer_inputs_are_built_as_llama_cpp_builds_them() {
    let (layers, width, n_embd) = (2usize, 2usize, 2usize);
    // Eighths: an unquantized table is held at f16, which these survive exactly.
    let table: Vec<f32> = (0..3 * layers * width)
        .map(|i| i as f32 * 0.125 - 0.5)
        .collect();
    let proj: Vec<f32> = vec![0.3, -0.2, 0.5, 0.1, -0.4, 0.6, 0.2, 0.7];
    let x0 = [1.5f32, -0.5];
    let q = |v: &[f32], shape: (usize, usize)| {
        QTensor::quantize(
            &Tensor::from_slice(v, shape, &Device::Cpu).unwrap(),
            GgmlDType::F32,
        )
        .unwrap()
    };
    let norm_w = [0.9f32, 1.2];
    let inputs = PerLayerInputs::new(
        q(&table, (3, layers * width)),
        QMatMul::from_qtensor(q(&proj, (layers * width, n_embd))).unwrap(),
        make_rms_norm(&Tensor::from_slice(&norm_w, (width,), &Device::Cpu).unwrap()),
        layers,
        width,
    )
    .unwrap();
    let ids = Tensor::new(&[[2u32]], &Device::Cpu).unwrap();
    let x = Tensor::from_slice(&x0, (1, 1, n_embd), &Device::Cpu).unwrap();
    let got = values(&inputs.forward(&ids, &x).unwrap());

    let mut want = Vec::new();
    for layer in 0..layers {
        let projected: Vec<f32> = (0..width)
            .map(|c| {
                let row = layer * width + c;
                (proj[row * n_embd] * x0[0] + proj[row * n_embd + 1] * x0[1])
                    / (n_embd as f32).sqrt()
            })
            .collect();
        let rms = (projected.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6).sqrt();
        for c in 0..width {
            let row = table[2 * layers * width + layer * width + c] * (width as f32).sqrt();
            want.push((projected[c] / rms * norm_w[c] + row) / 2f32.sqrt());
        }
    }
    assert!(
        largest_gap(&got, &want) < 1e-5,
        "got {got:?}, want {want:?}"
    );
}
