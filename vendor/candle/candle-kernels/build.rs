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
        .exclude(&["moe_*.cu", "ggml_mmq/*"]) // Exclude moe kernels for ptx build
        // SwarmLLM patch: llama.cpp's MMQ (`ggml_mmq/`) is a library, below.
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
    // SwarmLLM patch: a library carries its floor's PTX beside the machine
    // code. cudaforge's `build_lib` compiles `code=sm_NN` alone, and machine
    // code runs only within its own major version: a release built at 80 had
    // NO image of these kernels for a Hopper (9.0) or Blackwell (12.0) card,
    // which `gpu_support::compute_cap_supported` admits. The MoE launches
    // check no error, so their output was never written there. With the PTX
    // the driver compiles them for the card on first use, as it does
    // candle's own kernels (the PTX build above; gotcha #159). A newer card
    // is reproduced on any card by `CUDA_FORCE_PTX_JIT=1` — in a program
    // that loads nothing else: candle cannot open a device under it.
    let ptx_beside = format!("-gencode=arch=compute_{compute_cap},code=compute_{compute_cap}");
    moe_builder = moe_builder.arg(&ptx_beside);

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

    // SwarmLLM patch: llama.cpp's int8 tensor-core quantized matmul
    // (`src/ggml_mmq`: its ggml sources verbatim plus a launcher that takes
    // the caller's stream — see `src/ggml_mmq/swarm_mmq.cuh`). A library, as
    // the MoE kernels are: its templates are reached through C entry points
    // (`ffi.rs`), never loaded by name from PTX.
    let mmq_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("src/ggml_mmq");
    let mmq_includes: Vec<String> = ["include", "src", "src/ggml-cuda"]
        .iter()
        .map(|d| format!("-I{}", mmq_dir.join(d).display()))
        .collect();
    let mut mmq_builder = KernelBuilder::default()
        .source_dir("src/ggml_mmq")
        .watch(&headers)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg(&ptx_beside);
    for include in &mmq_includes {
        mmq_builder = mmq_builder.arg(include);
    }
    if is_target_msvc {
        mmq_builder = mmq_builder.arg("-D_USE_MATH_DEFINES");
    } else {
        mmq_builder = mmq_builder.arg("-Xcompiler").arg("-fPIC");
    }
    // SwarmLLM patch: `CANDLE_KERNELS_MMQ_CHECK_ONLY=1` skips this library and
    // still emits its link directives, as `CANDLE_FLASH_ATTN_CHECK_ONLY` does
    // for flash attention. CI's `cargo check` cells install `cuda-nvcc` alone,
    // and the copied `common.cuh` names cuBLAS types (`cublas_v2.h`): all
    // three failed on it (2026-10-21, gotcha #929). A check never links. A
    // build with it set must fail to link rather than ship without MMQ, so a
    // library an earlier build left here is deleted (`build_lib` archives
    // again when its file is missing, even with every object up to date).
    println!("cargo::rerun-if-env-changed=CANDLE_KERNELS_MMQ_CHECK_ONLY");
    let mmq_lib = out_dir.join("libggml_mmq.a");
    if env::var("CANDLE_KERNELS_MMQ_CHECK_ONLY").as_deref() == Ok("1") {
        println!(
            "cargo::warning=CANDLE_KERNELS_MMQ_CHECK_ONLY=1: llama.cpp's MMQ was NOT compiled. \
             This tree is valid for `cargo check` only; a build will fail to link."
        );
        let _ = std::fs::remove_file(&mmq_lib);
    } else {
        mmq_builder.build_lib(mmq_lib)?;
    }

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=ggml_mmq");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
    Ok(())
}

/// SwarmLLM patch: every header under `dir`, recursively — see `main`. `.h`
/// and `.inc` too: llama.cpp's ggml headers and its quantizer (`ggml_mmq/`).
fn cuda_headers(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("read the kernel sources").flatten() {
        let path = entry.path();
        if path.is_dir() {
            cuda_headers(&path, out);
        } else if path
            .extension()
            .is_some_and(|ext| ext == "cuh" || ext == "h" || ext == "inc")
        {
            out.push(path);
        }
    }
}
