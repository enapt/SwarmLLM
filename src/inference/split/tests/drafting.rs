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
        .draft_after(&kv, "fresh", 0, seq, gamma, &greedy(), &[], None, 0)
        .unwrap()
}

fn cache_len(model: &SplitModel, kv: &KvCacheStore, req: &str) -> usize {
    kv.get_or_create_keyed(&KvCacheStore::cache_key(model.kv_model_key(), req), 3)
        .layers[0]
        .as_ref()
        .map_or(0, |l| l.current_seq_len())
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
        .draft_after(&kv, "r", 0, &seq, 4, &greedy(), &[], None, 0)
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
        .draft_after(&kv, "r", valid, &seq[valid..], 3, &greedy(), &[], None, 0)
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
        .draft_after(&kv, "r", valid, &seq[valid..], 2, &greedy(), &[], None, 0)
        .unwrap();
    assert_eq!(d3, fresh(&mut model, &seq, 2));
    assert_eq!(cache_len(&model, &kv, "r"), seq.len() + 1);
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
        .draft_after(&kv, "n", 0, &seq, 3, &s, &[], Some(&noise), 0)
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
