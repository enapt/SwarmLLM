//! SwarmLLM Windows GPU launcher.
//!
//! Detects whether an NVIDIA GPU with drivers is present at runtime and
//! transparently executes the appropriate binary:
//!   - swarmllm-gpu.exe  — built with `windows-gpu` (Vulkan + static CUDA)
//!   - swarmllm-cpu.exe  — CPU-only fallback
//!
//! All CLI arguments are forwarded unchanged. Exit code is preserved.
//! This binary is renamed to `swarmllm.exe` in the Windows installer.

use std::env;
use std::path::PathBuf;
use std::process;

fn main() {
    let exe_dir = env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    let (binary_name, reason) = if has_nvidia_gpu() {
        ("swarmllm-gpu.exe", "NVIDIA GPU detected")
    } else {
        ("swarmllm-cpu.exe", "no NVIDIA GPU detected")
    };

    let mut binary_path = exe_dir.join(binary_name);

    // Fall back to CPU binary if GPU binary is missing (e.g. partial install).
    if !binary_path.exists() && binary_name == "swarmllm-gpu.exe" {
        eprintln!(
            "[SwarmLLM] {} not found, falling back to CPU binary",
            binary_path.display()
        );
        binary_path = exe_dir.join("swarmllm-cpu.exe");
    }

    if !binary_path.exists() {
        eprintln!(
            "[SwarmLLM] Could not find a SwarmLLM binary in {}. \
             Please reinstall.",
            exe_dir.display()
        );
        process::exit(1);
    }

    eprintln!("[SwarmLLM] {} — launching {}", reason, binary_name);

    let args: Vec<String> = env::args().skip(1).collect();

    let status = process::Command::new(&binary_path)
        .args(&args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!(
                "[SwarmLLM] Failed to launch {}: {}",
                binary_path.display(),
                e
            );
            process::exit(1);
        });

    process::exit(status.code().unwrap_or(1));
}

/// Returns true if an NVIDIA GPU with drivers is present.
///
/// Detection strategy:
/// - Windows: check for `nvcuda.dll` in System32 — present whenever NVIDIA
///   display drivers are installed (GeForce/Studio/Quadro/Data Center).
///   Does not require the CUDA Toolkit.
/// - Other platforms: check whether `nvidia-smi` runs successfully.
fn has_nvidia_gpu() -> bool {
    #[cfg(target_os = "windows")]
    {
        // nvcuda.dll lives in %WINDIR%\System32 on all NVIDIA driver installs.
        let windir = env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
        std::path::Path::new(&windir)
            .join("System32")
            .join("nvcuda.dll")
            .exists()
    }

    #[cfg(not(target_os = "windows"))]
    {
        process::Command::new("nvidia-smi")
            .arg("--query-gpu=name")
            .arg("--format=csv,noheader")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}
