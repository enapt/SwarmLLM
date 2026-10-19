//! A picture whose rows attend both ways (#115 (2b), llama.cpp's non-causal
//! image decode for Gemma 3) through a whole forward: what changes, what may
//! not, and how a chunked prompt pass keeps each picture in one chunk.

use super::common::*;
use crate::inference::split::kv_cache::KvCacheStore;
use candle_core::{Device, IndexOp, Tensor};

fn rows(t: &Tensor, range: std::ops::Range<usize>) -> Tensor {
    t.i((.., range, ..)).unwrap().contiguous().unwrap()
}

/// The mechanism fires, and only where it should: the rows inside the
/// picture differ from a causal pass (they now see the picture's later rows),
/// the text BEFORE it is untouched (it sees nothing new), and the text after
/// it differs (it reads the picture's changed rows).
#[test]
fn a_pictures_rows_see_each_other_and_the_text_before_it_is_untouched() {
    let hidden_dim = 64;
    let mut model = make_test_split_model(2, hidden_dim);
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 1.0, (1, 12, hidden_dim), &Device::Cpu).unwrap();
    let picture = 3..8;

    let causal = model.forward(&input, 0, &store, "causal").unwrap();
    let both_ways = model
        .with_image_spans(std::slice::from_ref(&picture), |m| {
            m.forward(&input, 0, &store, "both-ways")
        })
        .unwrap();

    assert_tensors_close(
        &rows(&both_ways, 0..3),
        &rows(&causal, 0..3),
        1e-6,
        "the text before the picture sees nothing it did not see before",
    );
    let differs = |a: &Tensor, b: &Tensor| {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            > 1e-4
    };
    assert!(
        differs(&rows(&both_ways, 3..7), &rows(&causal, 3..7)),
        "a picture's rows before its last see its later rows"
    );
    assert!(
        differs(&rows(&both_ways, 8..12), &rows(&causal, 8..12)),
        "the text after the picture reads the picture's changed rows"
    );
    assert!(
        model.image_spans.is_empty(),
        "the spans are forgotten after the call"
    );
}

/// A prompt pass in chunks gives the answer the whole pass gives when a
/// picture sits across a chunk boundary: the chunk runs on to the picture's
/// end. Without that, the picture's first rows would be computed before its
/// last ones exist.
#[test]
fn a_chunked_prompt_pass_keeps_each_picture_in_one_chunk() {
    let hidden_dim = 64;
    let mut model = make_test_split_model(2, hidden_dim);
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 1.0, (1, 14, hidden_dim), &Device::Cpu).unwrap();
    // Chunks of 4 would end at 4, 8 and 12: the first picture spans 4 → it
    // runs 0..7; the second spans 8 and 12 → it runs 7..13, then 13..14.
    let pictures = [2..7, 9..13];

    let whole = model
        .with_image_spans(&pictures, |m| m.forward(&input, 0, &store, "whole"))
        .unwrap();
    let chunked = model
        .with_image_spans(&pictures, |m| {
            m.forward_prompt_in_chunks(&input, 0, &store, "chunked", None, false, 4)
        })
        .unwrap();
    assert_tensors_close(
        &chunked,
        &whole,
        1e-4,
        "a chunked pass with pictures must return what the whole pass returns",
    );

    let next = Tensor::randn(0f32, 1.0, (1, 1, hidden_dim), &Device::Cpu).unwrap();
    let after_whole = model.forward(&next, 14, &store, "whole").unwrap();
    let after_chunked = model.forward(&next, 14, &store, "chunked").unwrap();
    assert_tensors_close(
        &after_chunked,
        &after_whole,
        1e-4,
        "and leave the same cache behind for the reply",
    );
}

/// A forward that would cut a picture in two is refused, never run with half
/// a picture's rows unable to see the other half.
#[test]
fn a_forward_that_cuts_a_picture_in_two_is_refused() {
    let hidden_dim = 64;
    let mut model = make_test_split_model(1, hidden_dim);
    let store = KvCacheStore::new(std::time::Duration::from_secs(600));
    let input = Tensor::randn(0f32, 1.0, (1, 5, hidden_dim), &Device::Cpu).unwrap();
    let err = model
        .with_image_spans(&[std::ops::Range { start: 3, end: 9 }], |m| {
            m.forward(&input, 0, &store, "cut")
        })
        .unwrap_err();
    assert!(err.to_string().contains("cuts the picture"), "{err}");
}
