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
    // SwarmLLM patch: an EMPTY value means unset.
    //
    // `std::env::var` returns `Ok("")` for a variable that is present but
    // empty, and CI sets exactly that — a matrix expression like
    // `${{ matrix.check_only && '...' || '' }}` yields an empty string for
    // every cell that does not want the override. Upstream's `Err(_)` arm does
    // not catch it, so the override branch would run on an empty path and
    // panic on every unrelated build. Filtering it here keeps the workflow
    // expression simple and matches how the sibling
    // `CANDLE_FLASH_ATTN_CHECK_ONLY` already behaves.
    let build_dir_override = std::env::var("CANDLE_FLASH_ATTN_BUILD_DIR")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let build_dir = match build_dir_override.ok_or(()) {
        Err(_) =>
        {
            #[allow(clippy::redundant_clone)]
            out_dir.clone()
        }
        // SwarmLLM patch: make an out-of-`target/` build dir usable on CI.
        //
        // This is the escape hatch that lets the 19 CUTLASS kernels survive
        // between runs. `Swatinem/rust-cache` deletes everything in `target/`
        // belonging to a package whose manifest lives inside the repo, and our
        // vendored crates do — so every GPU job restored a FULL cache hit and
        // then rebuilt all 19 kernels anyway: ~39 min of the 46-min Windows GPU
        // build and ~27 min of the Linux CUDA one, on every release.
        // `cudaforge`'s own `BuildCache` already knows how to skip them (it
        // compares CONTENT hashes, not mtimes, so a restored tarball is fine) —
        // it just needs its output to still be there.
        //
        // Two changes from upstream, both required for that to work:
        //
        //  * **Create the directory** rather than requiring it to pre-exist.
        //    Upstream panics with "Directory doesn't exists" on a cold cache,
        //    which is precisely the first run after this was introduced.
        //  * **Do not canonicalize an already-absolute path.** On Windows
        //    `canonicalize` returns an extended-length `\\?\C:\...` path, and
        //    that string is emitted verbatim as `cargo::rustc-link-search`.
        //    CI passes an absolute path, so there is nothing to resolve and
        //    nothing gained by risking the prefix. A relative path still gets
        //    canonicalized, which is what upstream's behaviour was for.
        Ok(build_dir) => {
            let path = PathBuf::from(build_dir);
            std::fs::create_dir_all(&path).unwrap_or_else(|e| {
                panic!(
                    "CANDLE_FLASH_ATTN_BUILD_DIR {} is not creatable: {e}",
                    path.display()
                )
            });
            if path.is_absolute() {
                path
            } else {
                path.canonicalize().unwrap_or_else(|e| {
                    panic!(
                        "CANDLE_FLASH_ATTN_BUILD_DIR {} could not be resolved ({e}); \
                         the current directory is {}",
                        path.display(),
                        std::env::current_dir().expect("cwd").display()
                    )
                })
            }
        }
    };

    // SwarmLLM patch: type-check-only mode.
    //
    // `cargo check` RUNS build scripts, so checking any feature that pulls this
    // crate compiles all 19 CUTLASS kernels — tens of minutes on a 4-vCPU CI
    // runner, since `thread_percentage(0.5)` below leaves it two threads. That
    // cost is why CI's feature-check matrix deliberately excluded the
    // `flash-attn` arm of `run_attention`, and why a missing import in that arm
    // broke every GPU build for five commits on 2026-08-07 with `cargo fmt`,
    // `cargo clippy --all-targets`, 1746 unit tests and the whole CI run green
    // (gotcha #264).
    //
    // With `CANDLE_FLASH_ATTN_CHECK_ONLY=1` the kernels are skipped and the
    // link directives are still emitted. `cargo check` never links, so the
    // type-check is complete and correct — it is exactly the Rust-side
    // coverage that was missing, in about a minute.
    //
    // A real `cargo build` with this set fails LOUDLY at link time, on a
    // missing `libflashattention.a`, rather than producing a binary with
    // silently absent kernels. That asymmetry is the point: the worst outcome
    // of a misuse is a failed build, not a shipped defect.
    println!("cargo::rerun-if-env-changed=CANDLE_FLASH_ATTN_CHECK_ONLY");
    let check_only = std::env::var("CANDLE_FLASH_ATTN_CHECK_ONLY").as_deref() == Ok("1");
    if check_only {
        println!(
            "cargo::warning=CANDLE_FLASH_ATTN_CHECK_ONLY=1: CUTLASS kernels were NOT compiled. \
             This tree is valid for `cargo check` only; a build will fail to link."
        );
        emit_link_directives(&build_dir, target_is_msvc());
        return Ok(());
    }

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
        // Upstream default: half the machine, so a build does not monopolise a
        // developer's workstation.
        //
        // SwarmLLM note (not a patch — nothing here changes): `cudaforge` reads
        // `CUDAFORGE_THREADS` and `RAYON_NUM_THREADS` AHEAD of this percentage,
        // so CI raises it to the full runner without editing this file. See
        // .github/actions/gpu-build-env. Anyone timing a cold kernel build
        // should check that variable before concluding this line is the cause.
        .thread_percentage(0.5);

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
    emit_link_directives(&build_dir, is_target_msvc);
    Ok(())
}

/// Whether the build target is MSVC. Needed before the builder is constructed
/// in the check-only path, which returns before `is_target_msvc` is computed.
fn target_is_msvc() -> bool {
    std::env::var("TARGET").is_ok_and(|t| t.contains("msvc"))
}

/// Tell rustc where the CUDA libraries live and what to link.
///
/// Shared by the normal path and the check-only path so the two cannot drift:
/// a `cargo check` that emitted different link metadata than a build would be
/// a check that does not check the thing being built.
fn emit_link_directives(build_dir: &std::path::Path, is_target_msvc: bool) {
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
}
