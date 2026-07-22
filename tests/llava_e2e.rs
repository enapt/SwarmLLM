//! End-to-end LLaVA multimodal run: image → vision embeddings → tokens.
//!
//! `#[ignore]` by design — it needs a ~4 GB text model and a ~600 MB mmproj
//! that are not committed, and it runs the 7B model on CPU because 7B plus 577
//! image tokens does not fit in 8 GB of VRAM. Expect minutes, not seconds.
//!
//! Run with:
//! ```text
//! SWARMLLM_TEST_LLAVA=/path/llava-v1.5-7b-Q4_K_M.gguf \
//! SWARMLLM_TEST_MMPROJ=/path/llava-v1.5-7b-mmproj-f16.gguf \
//! cargo test --test llava_e2e -- --ignored --nocapture
//! ```
//!
//! Covers the multimodal path that unit tests cannot reach: the real vicuna
//! prompt fallback, a real CLIP encode, and the `-1` marker splice in
//! `SplitModel::forward_multimodal` where vision embeddings replace the
//! `<image>` position (`split/executor.rs`).

use std::path::PathBuf;

use swarmllm::inference::split::SplitModel;
use swarmllm::types::{ChatMessage, ImageData, Role};

fn env_path(var: &str) -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var(var).ok()?);
    p.exists().then_some(p)
}

/// A solid square of `colour` on white.
///
/// The test runs two different colours and requires each answer to match its
/// own image. One colour proves nothing — "Red" is a plausible blind guess for
/// a model that never saw the pixels. Two, each answered correctly, cannot be
/// guessed.
fn colour_square(size: u32, colour: [u8; 3]) -> ImageData {
    let s = size as usize;
    let mut rgb = vec![255u8; 3 * s * s];
    for y in s / 4..3 * s / 4 {
        for x in s / 4..3 * s / 4 {
            let i = (y * s + x) * 3;
            rgb[i] = colour[0];
            rgb[i + 1] = colour[1];
            rgb[i + 2] = colour[2];
        }
    }
    ImageData {
        rgb_bytes: rgb,
        width: size,
        height: size,
    }
}

#[test]
#[ignore = "needs SWARMLLM_TEST_LLAVA + SWARMLLM_TEST_MMPROJ; minutes on CPU"]
fn llava_image_to_text() {
    let (Some(model_path), Some(mmproj_path)) = (
        env_path("SWARMLLM_TEST_LLAVA"),
        env_path("SWARMLLM_TEST_MMPROJ"),
    ) else {
        eprintln!("SKIP: set SWARMLLM_TEST_LLAVA and SWARMLLM_TEST_MMPROJ to real GGUF paths");
        return;
    };

    let device = candle_core::Device::Cpu;

    for (colour, rgb, wrong) in [
        ("red", [255u8, 0, 0], "blue"),
        ("blue", [0u8, 0, 255], "red"),
    ] {
        eprintln!("\n───────── {colour} square ─────────");
        run_one(&model_path, &mmproj_path, &device, colour, rgb, wrong);
    }
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    model_path: &std::path::Path,
    mmproj_path: &std::path::Path,
    device: &candle_core::Device,
    colour: &str,
    rgb: [u8; 3],
    wrong: &str,
) {
    // ── 1. Prompt construction, through the production builder ──
    //
    // This GGUF embeds no `tokenizer.chat_template`, so `None` here is what
    // the daemon really passes; the model-name heuristic selects vicuna.
    let messages = vec![ChatMessage {
        role: Role::User,
        content: "What color is the shape in this image?".into(),
        images: vec![colour_square(336, rgb)],
    }];
    let prompt = swarmllm::inference::chat_template::build_prompt_with_model(
        &messages,
        None,
        "<s>",
        "</s>",
        Some("llava-v1.5-7b-Q4_K_M"),
    );
    eprintln!("prompt: {prompt:?}");
    assert!(
        prompt.contains("USER:") && prompt.contains("ASSISTANT:"),
        "expected vicuna framing, got: {prompt:?}"
    );
    assert!(
        prompt.contains("<image>"),
        "image placeholder missing — vision embeddings would land at the wrong \
         position: {prompt:?}"
    );

    // ── 2. Vision tower ──
    eprintln!("loading mmproj…");
    let t = std::time::Instant::now();
    let vision = swarmllm::inference::vision::load_from_mmproj_gguf(mmproj_path, device)
        .expect("load mmproj");
    let vision_emb = vision
        .encode_images(std::slice::from_ref(&messages[0].images[0]))
        .expect("encode image");
    eprintln!(
        "vision: {:?} in {:.1}s",
        vision_emb.dims(),
        t.elapsed().as_secs_f64()
    );

    // ── 3. Text model, whole GGUF as one segment ──
    eprintln!("loading text model (this is the slow part)…");
    let t = std::time::Instant::now();
    let n_layers = 32;
    let mut model = SplitModel::load_from_gguf(model_path, 0, n_layers, true, true, true)
        .expect("load text model");
    eprintln!("model loaded in {:.1}s", t.elapsed().as_secs_f64());

    // ── 4. Splice the marker ──
    //
    // Mirrors `split/executor.rs::forward_multimodal`: tokens before `<image>`,
    // a single `-1` standing in for it, then tokens after. The encoder replaces
    // that one position with all 577 vision embeddings.
    let (head, tail) = prompt.split_once("<image>").expect("placeholder present");
    let mut ids: Vec<i64> = model
        .encode_ids(head)
        .into_iter()
        .map(|t| t as i64)
        .collect();
    ids.push(-1);
    ids.extend(model.encode_ids(tail).into_iter().map(|t| t as i64));
    eprintln!(
        "{} prompt tokens (marker at {:?})",
        ids.len(),
        ids.iter().position(|&t| t == -1)
    );

    let kv = swarmllm::inference::split::KvCacheStore::new(std::time::Duration::from_secs(600));
    let req = "llava-e2e";

    // ── 5. Prefill + greedy decode ──
    let t = std::time::Instant::now();
    let input = candle_core::Tensor::new(ids.as_slice(), device)
        .and_then(|t| t.unsqueeze(0))
        .expect("input tensor");
    let mut logits = model
        .forward_multimodal(&input, 0, &kv, req, Some(&vision_emb))
        .expect("prefill");
    // 1 marker token expands to N vision tokens, so the next position is
    // offset by the expansion, not by ids.len(). Getting this wrong does not
    // error — it silently misaligns the KV cache and yields fluent nonsense.
    let base_pos = ids.len() - 1 + vision_emb.dims()[0];
    eprintln!(
        "prefill done in {:.1}s, next index_pos={base_pos}",
        t.elapsed().as_secs_f64()
    );

    let eos: Vec<u32> = model.eos_tokens().to_vec();
    let vocab = model.vocab().cloned().unwrap_or_default();
    let mut out = String::new();

    for step in 0..24 {
        let mut flat: Vec<f32> = logits
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .expect("logits to vec");
        let next = flat
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .expect("argmax");
        if eos.contains(&next) {
            eprintln!("hit EOS at step {step}");
            break;
        }
        if let Some(s) = vocab.get(next as usize) {
            out.push_str(&s.replace('\u{2581}', " "));
        }
        flat.clear();

        let step_in = candle_core::Tensor::new(&[next as i64][..], device)
            .and_then(|t| t.unsqueeze(0))
            .expect("step tensor");
        logits = model
            .forward_multimodal(&step_in, base_pos + step, &kv, req, None)
            .expect("decode step");
    }

    eprintln!("\n=== generated ===\n{}\n=================", out.trim());
    // The real check: the answer must name THIS image's colour and not the
    // other run's. A model that never saw the pixels cannot get both right.
    let lower = out.to_lowercase();
    assert!(
        lower.contains(colour),
        "expected the answer to mention {colour:?}, got {out:?}"
    );
    assert!(
        !lower.contains(wrong),
        "answer mentioned {wrong:?} for a {colour} image — vision embeddings are \
         not reaching the model correctly: {out:?}"
    );
}
