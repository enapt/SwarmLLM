use std::path::PathBuf;

fn main() {
    // Frontend rebuild tracking moved to crates/swarmllm-frontend/build.rs
    //
    // CUDA library search paths are NOT set here. `candle-flash-attn` is the
    // crate that declares the link (`rustc-link-lib=static=cudart_static`), and
    // rustc resolves a `static=` library while building THAT crate's rlib — a
    // search path emitted from this package arrives too late and fails with
    // `could not find native static library cudart_static`. It lives in
    // vendor/candle-flash-attn/build.rs, whose directives propagate to the
    // final binary link as well.

    build_fused_decode_ptx();
}

/// Compile `kernels/*.cu` to PTX for the `candle-cuda` builds.
///
/// The fused decode kernels remove a launch, an allocation and a free per layer
/// each — the lever that matters on a decoder bound by submission count rather
/// than bandwidth (`docs/invariants/inference.md`). They load at runtime
/// through candle's `CudaDevice::get_or_load_custom_func`, which takes PTX as a
/// string, so this is the whole build-side story: no fifth vendored crate, and
/// `candle-kernels` stays a registry dependency.
///
/// PTX rather than cubin on purpose — PTX JITs forward onto any card at or
/// above the virtual arch, so one artifact serves the whole `sm_80+` range the
/// CUDA release builds target.
fn build_fused_decode_ptx() {
    const KERNELS: &[&str] = &["fused_decode.cu"];

    for k in KERNELS {
        println!("cargo:rerun-if-changed=kernels/{k}");
    }
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");

    // Nothing to build for a CPU-only binary, which is every default build and
    // all of CI's fast lanes.
    if std::env::var_os("CARGO_FEATURE_CANDLE_CUDA").is_none() {
        return;
    }

    // ⚠ An empty env var is not an unset one. The release and cache-warm
    // matrices set `CUDA_COMPUTE_CAP: ${{ ... && '80' || '' }}`, which hands
    // every non-CUDA cell `Ok("")` — the trap `arch-inference.md` records
    // against the flash-attn build dir. Filter it explicitly.
    let compute_cap = std::env::var("CUDA_COMPUTE_CAP")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "80".to_string());

    let nvcc = match std::env::var_os("CUDA_ROOT") {
        Some(root) => PathBuf::from(root).join("bin").join("nvcc"),
        None => {
            let default = PathBuf::from("/usr/local/cuda/bin/nvcc");
            if default.exists() {
                default
            } else {
                PathBuf::from("nvcc")
            }
        }
    };

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

    for k in KERNELS {
        let src = PathBuf::from("kernels").join(k);
        let dst = out_dir.join(k).with_extension("ptx");

        // `-O3 -std=c++17` matches what `candle-kernels` builds its own
        // kernels with (its build.rs, via cudaforge), and so does the ABSENCE
        // of `-use_fast_math` — which is the load-bearing one. Each kernel
        // here must stay bit-identical to the candle ops it replaces, so that
        // swapping between them measures the submission count and not the
        // arithmetic; fast math would make `expf` a different function on one
        // side of the A/B. See the header comment in
        // `kernels/fused_decode.cu`.
        let out = std::process::Command::new(&nvcc)
            .arg("--ptx")
            .arg(format!("-arch=compute_{compute_cap}"))
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-o")
            .arg(&dst)
            .arg(&src)
            .output();

        match out {
            Ok(o) if o.status.success() => {}
            // Fail the build rather than shipping a `candle-cuda` binary whose
            // fused path silently does not exist. Every feature set that turns
            // `candle-cuda` on also builds candle-flash-attn's CUTLASS
            // kernels, so a toolkit is already a hard requirement here — an
            // absent nvcc is a broken environment, not a configuration.
            Ok(o) => panic!(
                "nvcc failed to compile kernels/{k} to PTX (status {}):\n{}\n{}",
                o.status,
                String::from_utf8_lossy(&o.stdout),
                String::from_utf8_lossy(&o.stderr),
            ),
            Err(e) => panic!(
                "could not run {} to compile kernels/{k}: {e}\n\
                 Set CUDA_ROOT to the toolkit root if nvcc is installed elsewhere.",
                nvcc.display()
            ),
        }
    }
}
