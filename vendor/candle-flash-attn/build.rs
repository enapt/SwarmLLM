// Build script to run nvcc and generate the C glue code for launching the flash-attention kernel.
// The cuda build time is very long so one can set the CANDLE_FLASH_ATTN_BUILD_DIR environment
// variable in order to cache the compiled artifacts and avoid recompiling too often.
use cudaforge::{KernelBuilder, Result};
use std::path::PathBuf;
const CUTLASS_COMMIT: &str = "7d49e6c7e2f8896c47f586706e67e1fb215529dc";

// SwarmLLM patch: the 18 `_bf16_` kernels are removed.
//
// `run_attention`'s CUDA branch casts Q/K/V to F16 before every call
// (src/inference/layers/mod.rs), so the bf16 half of this matrix has no
// reachable caller in this binary — it was half the kernel compile for
// nothing. `FlashAttn::cuda_fwd`'s BF16 dispatch arm is removed to match,
// so a bf16 tensor gets a clear error rather than a link failure.
//
// Why it matters: this build script asks for only HALF the machine's threads
// (`thread_percentage(0.5)`, below) and each CUTLASS kernel is expensive.
// Measured at 8-way parallelism on 16 cores, the full 37 took ~45 min; on a
// 4-vCPU CI runner that projects to ~2.5 h, which is what got flash-attn
// dropped from the default feature set in 2026-04 in the first place.
const KERNEL_FILES: [&str; 19] = [
    "kernels/flash_api.cu",
    "kernels/flash_fwd_hdim128_fp16_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_sm80.cu",
    "kernels/flash_fwd_hdim128_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim160_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim192_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim224_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim256_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim512_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim32_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim64_fp16_causal_sm80.cu",
    "kernels/flash_fwd_hdim96_fp16_causal_sm80.cu",
];

fn main() -> Result<()> {
    println!("cargo::rerun-if-changed=build.rs");
    for kernel_file in KERNEL_FILES.iter() {
        println!("cargo::rerun-if-changed={kernel_file}");
    }
    println!("cargo::rerun-if-changed=kernels/flash_fwd_kernel.h");
    println!("cargo::rerun-if-changed=kernels/flash_fwd_launch_template.h");
    println!("cargo::rerun-if-changed=kernels/flash.h");
    println!("cargo::rerun-if-changed=kernels/philox.cuh");
    println!("cargo::rerun-if-changed=kernels/softmax.h");
    println!("cargo::rerun-if-changed=kernels/utils.h");
    println!("cargo::rerun-if-changed=kernels/kernel_traits.h");
    println!("cargo::rerun-if-changed=kernels/block_info.h");
    println!("cargo::rerun-if-changed=kernels/static_switch.h");
    println!("cargo::rerun-if-changed=kernels/hardware_info.h");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let build_dir = match std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR") {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            path.canonicalize().expect(&format!(
                "Directory doesn't exists: {} (the current directory is {})",
                &path.display(),
                std::env::current_dir()?.display()
            ))
        }
    };

    let kernels: Vec<_> = KERNEL_FILES.iter().collect();
    let mut builder = KernelBuilder::new()
        .source_files(kernels)
        .out_dir(&build_dir)
        .with_cutlass(Some(CUTLASS_COMMIT)) // ✅ Auto-fetch and include CUTLASS from GitHub
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-U__CUDA_NO_HALF_OPERATORS__")
        .arg("-U__CUDA_NO_HALF_CONVERSIONS__")
        .arg("-U__CUDA_NO_HALF2_OPERATORS__")
        .arg("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
        .arg("--expt-relaxed-constexpr")
        .arg("--expt-extended-lambda")
        .arg("--use_fast_math")
        .arg("--verbose")
        .thread_percentage(0.5); // Use up to 50% of available threads

    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            builder = builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        builder = builder.arg("-Xcompiler").arg("-fPIC");
    }

    let out_file = build_dir.join("libflashattention.a");
    builder.build_lib(out_file)?;

    // SwarmLLM patch: tell rustc where the CUDA libraries live.
    //
    // Upstream emits `rustc-link-lib` with no matching `rustc-link-search`,
    // relying on the toolkit's lib dir already being on the linker's default
    // path. It is not, on this project's CI (`/usr/local/cuda-12.8/lib64`) or on
    // a typical developer box, and the failure is a bare
    // `unable to find library -lcudart` at link time.
    //
    // It has to be emitted HERE rather than from the consuming package: rustc
    // resolves a `static=` link library while building THIS crate's rlib, so a
    // search path added downstream arrives too late
    // (`could not find native static library cudart_static`). Build-script
    // link-search directives still propagate to the final binary link, so this
    // one placement covers both.
    //
    // CUDA_PATH is what the CI action exports and what the NVIDIA installers
    // set; /usr/local/cuda is the conventional symlink. `lib/x64` is Windows.
    println!("cargo::rerun-if-env-changed=CUDA_PATH");
    println!("cargo::rerun-if-env-changed=CUDA_HOME");
    for root in [
        std::env::var("CUDA_PATH").ok(),
        std::env::var("CUDA_HOME").ok(),
        Some("/usr/local/cuda".to_string()),
    ]
    .into_iter()
    .flatten()
    {
        for sub in ["lib64", "lib/x64", "lib"] {
            let dir = PathBuf::from(&root).join(sub);
            if dir.is_dir() {
                println!("cargo::rustc-link-search=native={}", dir.display());
            }
        }
    }

    println!("cargo::rustc-link-search={}", build_dir.display());
    println!("cargo::rustc-link-lib=flashattention");
    // SwarmLLM patch: static CUDA runtime instead of `dylib=cudart`.
    //
    // Upstream links the runtime dynamically, which puts a hard
    // `libcudart.so.N` / `cudart64_N.dll` dependency on the final binary.
    // This project deliberately avoids that: `cudarc` is configured with
    // `dynamic-loading` so a release binary needs ONLY the NVIDIA display
    // driver (`libcuda.so.1`), which every machine with an NVIDIA GPU already
    // has — see the annotated rationale on the `cudarc` dependency in the root
    // Cargo.toml. The shipped v0.3.81 Linux CUDA binary's `ldd` output is a
    // single CUDA line, `libcuda.so.1`, and this keeps it that way.
    //
    // Linking dynamically here would have turned a soft failure (no CUDA
    // runtime present -> dlopen fails -> fall back) into the binary refusing
    // to exec at all, which is the worse outcome for a non-technical audience.
    //
    // `cudart_static` needs `culibos` for its thread/rt shims on Linux; MSVC
    // links its own equivalents, so it must not be requested there.
    println!("cargo::rustc-link-lib=static=cudart_static");
    if !is_target_msvc {
        println!("cargo::rustc-link-lib=dylib=culibos");
    }
    if !is_target_msvc {
        println!("cargo::rustc-link-lib=dylib=stdc++");
    }
    Ok(())
}
