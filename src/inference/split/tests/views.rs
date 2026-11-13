//! A loaded model serves any run of its layers from its own weights (#449,
//! `split::view`).

use super::super::LayerVariant;
use super::common::make_test_split_model;
use crate::inference::split::kv_cache::KvCacheStore;
use crate::inference::split::view::shares_weights_with;
use candle_core::{Device, Tensor};

fn store() -> KvCacheStore {
    KvCacheStore::new(std::time::Duration::from_secs(600))
}

/// A view of layers 1-2 of a loaded 0-3 runs THOSE layers — the loaded
/// copy's own weights, not a copy and not its first two — and answers bit for
/// bit what the loaded copy answers with its other layers passed over, for a
/// prompt and for a step on the cache that prompt left.
#[test]
fn a_view_runs_its_layers_of_the_loaded_copy() {
    let hidden = 64;
    let mut parent = make_test_split_model(4, hidden);
    let mut view = parent.view_of(1, 3).expect("a run inside the loaded range");
    assert_eq!((view.layer_start, view.layer_end), (1, 3));
    assert_eq!(view.layers.len(), 2);
    assert_eq!(view.viewed_from(), Some((0, 4)));
    assert!(shares_weights_with(&view, &parent), "the loaded weights");
    // The fixture is a middle segment (`total_layers` past its end), so the
    // view keeps the model's total in its key.
    assert_eq!(
        view.kv_model_key(),
        format!("1-3-{}", parent.total_layers),
        "a cache of its own"
    );

    let skip = [true, false, false, true];
    let (parents, views) = (store(), store());
    let prompt = Tensor::randn(0f32, 1.0, (1, 3, hidden), &Device::Cpu).unwrap();
    let want = parent
        .forward_with_skip_mask(&prompt, 0, &parents, "r", &skip)
        .unwrap();
    let got = view.forward(&prompt, 0, &views, "r").unwrap();
    assert_eq!(
        want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "the prompt pass"
    );
    let step = Tensor::randn(0f32, 1.0, (1, 1, hidden), &Device::Cpu).unwrap();
    let want = parent
        .forward_with_skip_mask(&step, 3, &parents, "r", &skip)
        .unwrap();
    let got = view.forward(&step, 3, &views, "r").unwrap();
    assert_eq!(
        want.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        got.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        "a step on the cache the prompt left"
    );

    // The loaded copy's first two layers answer differently: the view ran
    // the right ones, not the first ones.
    let mut first_two = parent.view_of(0, 2).unwrap();
    let other = first_two.forward(&prompt, 0, &store(), "r").unwrap();
    assert_ne!(
        other.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        view.forward(&prompt, 0, &store(), "q")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    );
}

/// What a view is never made of: the range itself, a range reaching outside
/// it, an empty one — and a model whose layers read each other's caches,
/// which runs only whole.
#[test]
fn a_view_is_only_a_run_inside_a_model_that_may_be_cut() {
    let mut parent = make_test_split_model(4, 64);
    assert!(parent.view_of(0, 4).is_none(), "the range itself");
    assert!(parent.view_of(2, 5).is_none(), "outside it");
    assert!(parent.view_of(2, 2).is_none(), "empty");
    assert!(parent.view_of(0, 1).is_some());
    if let LayerVariant::Dense(w) = &mut parent.layers[3] {
        w.kv_source = Some(1);
    }
    assert!(
        parent.view_of(0, 2).is_none(),
        "a layer lending or borrowing a cache runs only whole"
    );
}
