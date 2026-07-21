# Installation

## Download

Download the right file for your system from the [GitHub Releases page](https://github.com/enapt/SwarmLLM/releases/latest):

| Your Computer | File Name |
|---|---|
| **Windows** (most PCs) | `SwarmLLM-Setup.exe` (installer — auto-detects GPU) |
| **Windows** (raw binary, GPU) | `swarmllm-windows-x86_64-gpu.zip` |
| **Windows** (raw binary, CPU) | `swarmllm-windows-x86_64-cpu.zip` |
| **Mac** (M1/M2/M3/M4) | `swarmllm-macos-aarch64.tar.gz` (compile-validated) |
| **Mac** (older Intel) | Best-effort — build from source |
| **Linux** (most distros) | `swarmllm-linux-x86_64.tar.gz` |
| **Linux** (NVIDIA GPU) | `swarmllm-linux-x86_64-cuda.tar.gz` |

> **Not sure which Mac?** Apple menu > "About This Mac." If it says "Apple M1" (or M2/M3/etc.), pick Apple Silicon. If it says "Intel," pick Intel.

## Install & Run

### Windows

**Recommended — installer:** double-click `SwarmLLM-Setup.exe`. It detects your GPU (NVIDIA / AMD / Intel) and installs the matching binary. If SmartScreen warns you, click **More info** > **Run anyway**.

**Raw binary alternative:** download `swarmllm-windows-x86_64-gpu.zip` (Vulkan + CUDA static) or `swarmllm-windows-x86_64-cpu.zip` (CPU-only fallback), extract, and run `swarmllm.exe`.

From PowerShell on a raw binary:
```powershell
cd Downloads\swarmllm-windows-x86_64-gpu
.\swarmllm.exe run
```

### macOS

```bash
cd ~/Downloads
tar xzf swarmllm-macos-aarch64.tar.gz
cd swarmllm-macos-aarch64
chmod +x swarmllm
./swarmllm run
```

> **Note:** macOS aarch64 binaries are compile-validated and exercised in CI (test + clippy on `macos-15`); integration tests stay Linux-only for now. Intel Mac users should build from source. If macOS blocks the binary on first launch: System Settings > Privacy & Security > click **Open Anyway** next to SwarmLLM.

### Linux

```bash
cd ~/Downloads
tar xzf swarmllm-linux-x86_64.tar.gz
cd swarmllm-linux-x86_64
chmod +x swarmllm
./swarmllm run
```

### Docker

The fastest way to get running on any Linux server:

```bash
# 1. Get the compose file and example env
curl -LO https://raw.githubusercontent.com/enapt/SwarmLLM/main/docker-compose.yml
curl -LO https://raw.githubusercontent.com/enapt/SwarmLLM/main/.env.example

# 2. Configure (add API keys, change ports, etc.)
cp .env.example .env
nano .env

# 3. Start
docker compose up -d
```

For NVIDIA GPU support (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
docker compose --profile gpu up -d
```

Pre-built images on GHCR:

| Image | Description |
|---|---|
| `ghcr.io/enapt/swarmllm:latest` | CPU-only |
| `ghcr.io/enapt/swarmllm:latest-cuda` | NVIDIA GPU (CUDA 12.4) |
| `ghcr.io/enapt/swarmllm:0.3.4-alpha` | Pinned version (CPU) |
| `ghcr.io/enapt/swarmllm:0.3.4-alpha-cuda` | Pinned version (GPU) |

Data is persisted in Docker volumes. Model shards are stored in the `swarmllm-models` volume (or bind-mount a host directory via `SWARMLLM_MODELS_DIR` in `.env`).

View logs with `docker compose logs -f`. The API key is printed on first startup.

### Cargo Install

Requires Rust 1.80+:

```bash
cargo install --git https://github.com/enapt/SwarmLLM.git --tag v0.3.4-alpha
swarmllm run
```

### Building from Source

```bash
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM
cargo build --release
./target/release/swarmllm run
```

For CUDA GPU support:
```bash
cargo build --release --features candle-cuda
```

For Apple Silicon: the default build runs on CPU. A Metal-accelerated
build is on the roadmap but not yet implemented (no `metal` Cargo
feature exists yet); until then, use the default `cargo build --release`.

## Open the Dashboard

Once running, open **[http://localhost:8800](http://localhost:8800)** in your browser. The setup wizard will walk you through initial configuration.
