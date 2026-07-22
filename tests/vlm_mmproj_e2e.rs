// Standalone test: load a real mmproj GGUF and encode an image.
//
// Run with: cargo test --test vlm_mmproj_e2e -- --nocapture
//
// Point it at a file with SWARMLLM_TEST_MMPROJ, otherwise it looks in the
// platform data dir. Skips (does not fail) when no mmproj is present, since
// the file is ~600 MB and not committed.
//
// What this covers that unit tests cannot: `load_from_mmproj_gguf` resolves
// the CLIP FFN tensor naming by shape (see `vision::clip_ffn_is_swapped`).
// Getting that backwards is not silent — the first MLP matmul fails on a
// dimension mismatch — so a successful encode here is the real check that
// the detection agrees with an actual file.

use std::path::PathBuf;

/// Where to look for an mmproj, in order.
fn find_mmproj() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SWARMLLM_TEST_MMPROJ") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let mut roots = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(&home).join(".local/share/swarmllm/mmproj"));
        roots.push(PathBuf::from(&home).join("Library/Application Support/swarmllm/mmproj"));
    }
    if let Some(appdata) = std::env::var_os("APPDATA") {
        roots.push(PathBuf::from(appdata).join("swarmllm/mmproj"));
    }
    roots.push(PathBuf::from("/tmp/vlm_test"));

    for dir in roots {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "gguf") {
                return Some(p);
            }
        }
    }
    None
}

#[test]
fn vlm_mmproj_load_and_encode() {
    let Some(mmproj_path) = find_mmproj() else {
        eprintln!(
            "SKIP: no mmproj .gguf found. Set SWARMLLM_TEST_MMPROJ=/path/to/mmproj.gguf to run."
        );
        return;
    };
    let mmproj_path = mmproj_path.as_path();

    let device = candle_core::Device::Cpu;

    eprintln!("Loading mmproj from {}...", mmproj_path.display());
    let start = std::time::Instant::now();
    let vision_module = swarmllm::inference::vision::load_from_mmproj_gguf(mmproj_path, &device)
        .expect("Failed to load mmproj");
    eprintln!("Loaded in {:.1}s", start.elapsed().as_secs_f64());

    eprintln!(
        "Config: image_size={}, patch_size={}, hidden_size={}, num_heads={}, num_layers={}, projection_dim={}",
        vision_module.encoder.config().image_size,
        vision_module.encoder.config().patch_size,
        vision_module.encoder.config().vision_hidden_size,
        vision_module.encoder.config().vision_num_heads,
        vision_module.encoder.config().vision_num_layers,
        vision_module.encoder.config().projection_dim,
    );
    eprintln!("Tokens per image: {}", vision_module.num_image_tokens());

    // Create a test image (red square on white background)
    let size = vision_module.encoder.config().image_size;
    let s = size as usize;
    let mut rgb = vec![255u8; 3 * s * s];
    for y in s / 4..3 * s / 4 {
        for x in s / 4..3 * s / 4 {
            let idx = (y * s + x) * 3;
            rgb[idx] = 255; // R
            rgb[idx + 1] = 0; // G
            rgb[idx + 2] = 0; // B
        }
    }

    let image = swarmllm::types::ImageData {
        rgb_bytes: rgb,
        width: size,
        height: size,
    };

    eprintln!("Encoding image ({size}x{size})...");
    let start = std::time::Instant::now();
    let embeddings = vision_module
        .encode_images(&[image])
        .expect("Failed to encode image");
    eprintln!(
        "Encoded in {:.1}s, shape: {:?}",
        start.elapsed().as_secs_f64(),
        embeddings.dims()
    );

    let num_tokens = vision_module.num_image_tokens();
    let llm_hidden = vision_module.projection.llm_hidden_dim();
    assert_eq!(
        embeddings.dims(),
        &[num_tokens, llm_hidden],
        "Expected ({num_tokens}, {llm_hidden}), got {:?}",
        embeddings.dims()
    );

    // Verify embeddings are not all zeros
    let flat: Vec<f32> = embeddings.flatten_all().unwrap().to_vec1().unwrap();
    let mean = flat.iter().sum::<f32>() / flat.len() as f32;
    let max_abs = flat.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!("Embedding stats: mean={mean:.6}, max_abs={max_abs:.4}");
    assert!(max_abs > 0.001, "Embeddings should not be all zeros");

    eprintln!("VLM mmproj E2E test PASSED");
}
