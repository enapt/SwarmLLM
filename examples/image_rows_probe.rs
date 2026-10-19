//! This node's reading of a picture, compared with llama.cpp's (FUTURE_WORK #310).
//!
//! ```text
//! cargo run --release --no-default-features --features dev --example image_rows_probe -- \
//!     MMPROJ.gguf IMAGE [REFERENCE.f32 [OURS.f32]]
//! ```
//!
//! Encodes IMAGE with the image reader in MMPROJ on the processor — the rows a
//! request's picture becomes — and, given the rows `examples/image_rows_reference.cpp`
//! wrote for the same picture, prints how far apart they are: the row counts, each
//! row's cosine with its reference row (worst, median), and the relative error of
//! all of them together. A correct reader is at cosine ~0.9999 everywhere; a
//! different reading of the picture (an extra row, another activation, another
//! resize) is well below it. Where the counts differ by one, the extra leading row
//! is set aside so the rest can still be compared (it is reported). OURS.f32
//! receives this node's rows in the reference's format, to look at row by row.
//!
//! `SWARMLLM_LLAVA_DEPTH=llama-cpp` reads a LLaVA picture to llama.cpp's depth
//! (one encoder layer fewer — see `vision::encoder_depth`), which is how the rest
//! of the reader is checked against llama.cpp on equal terms.

use swarmllm::types::ImageData;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: {} MMPROJ.gguf IMAGE [REFERENCE.f32 [OURS.f32]]",
            args[0]
        );
        std::process::exit(2);
    }
    let device = candle_core::Device::Cpu;
    let module = swarmllm::inference::vision::load_from_mmproj_gguf(args[1].as_ref(), &device)
        .unwrap_or_else(|e| panic!("load {}: {e}", args[1]));
    let img = image::open(&args[2])
        .unwrap_or_else(|e| panic!("read {}: {e}", args[2]))
        .to_rgb8();
    let image = ImageData {
        width: img.width(),
        height: img.height(),
        rgb_bytes: img.into_raw(),
    };
    let start = std::time::Instant::now();
    let rows = module
        .encode_images(std::slice::from_ref(&image))
        .expect("encode");
    let (n, width) = rows.dims2().expect("two dims");
    let ours: Vec<f32> = rows
        .flatten_all()
        .and_then(|t| t.to_vec1())
        .expect("rows to host");
    println!(
        "ours: rows={n} width={width} image={}x{} in {:.1}s",
        image.width,
        image.height,
        start.elapsed().as_secs_f64()
    );

    if let Some(out) = args.get(4) {
        let bytes: Vec<u8> = ours.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out, bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
    }
    let Some(reference) = args.get(3) else { return };
    let bytes = std::fs::read(reference).unwrap_or_else(|e| panic!("read {reference}: {e}"));
    let theirs: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let their_rows = theirs.len() / width;
    println!("reference: rows={their_rows}");
    let skip = match n.checked_sub(their_rows) {
        Some(0) => 0,
        Some(1) => {
            println!("ours has ONE more row; comparing from our second row on");
            1
        }
        _ => {
            println!("row counts differ by more than one; nothing to compare");
            std::process::exit(1);
        }
    };
    let mut cosines = Vec::with_capacity(their_rows);
    let (mut diff2, mut ref2) = (0f64, 0f64);
    for r in 0..their_rows {
        let a = &ours[(r + skip) * width..(r + skip + 1) * width];
        let b = &theirs[r * width..(r + 1) * width];
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (*x as f64, *y as f64);
            dot += x * y;
            na += x * x;
            nb += y * y;
            diff2 += (x - y) * (x - y);
        }
        ref2 += nb;
        cosines.push(dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE));
    }
    let worst = cosines
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, c)| (i, *c))
        .unwrap_or((0, f64::NAN));
    let mut sorted = cosines.clone();
    sorted.sort_by(f64::total_cmp);
    println!(
        "cosine: worst {:.6} (row {}), median {:.6}; relative error {:.5}",
        worst.1,
        worst.0,
        sorted[sorted.len() / 2],
        (diff2 / ref2).sqrt()
    );
}
