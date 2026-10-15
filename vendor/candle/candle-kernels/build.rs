use cudaforge::{KernelBuilder, Result};
use std::env;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    // SwarmLLM patch: every header under src/ is WATCHED by the kernel cache
    // (cudaforge also tells cargo). An object is rebuilt when its `.cu` file,
    // the nvcc arguments or a watched path changes, and upstream watched no
    // header — so a fix in one (`moe/gguf.cuh`, say) was reported "up-to-date"
    // and never reached the binary. Found when the same hole in
    // vendor/candle-flash-attn/build.rs kept a fixed flash kernel out
    // (FUTURE_WORK #115 (3)). Upstream's hand list named three of the five.
    let mut headers = Vec::new();
    cuda_headers(std::path::Path::new("src"), &mut headers);
    headers.sort();

    // Build for PTX
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let bindings = KernelBuilder::new()
        .source_dir("src") // Scan src/ for .cu files
        .watch(&headers)
        .exclude(&["moe_*.cu"]) // Exclude moe kernels for ptx build
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .build_ptx()?;

    bindings.write(&ptx_path)?;

    let mut moe_builder = KernelBuilder::default()
        .source_files(vec![
            "src/moe/moe_gguf.cu",
            "src/moe/moe_wmma.cu",
            "src/moe/moe_wmma_gguf.cu",
        ])
        .watch(&headers)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Disable bf16 WMMA kernels on GPUs older than sm_80 (Ampere).
    // bf16 WMMA fragments require compute capability >= 8.0.
    let compute_cap = cudaforge::detect_compute_cap()
        .map(|arch| arch.base())
        .unwrap_or(80);
    if compute_cap < 80 {
        moe_builder = moe_builder.arg("-DNO_BF16_KERNEL");
    }

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    moe_builder.build_lib(out_dir.join("libmoe.a"))?;
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}

/// SwarmLLM patch: every `.cuh` under `dir`, recursively — see `main`.
fn cuda_headers(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read the kernel sources").flatten() {
        let path = entry.path();
        if path.is_dir() {
            cuda_headers(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "cuh") {
            out.push(path);
        }
    }
}
