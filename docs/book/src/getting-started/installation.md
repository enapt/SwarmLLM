# Installation

## Download

Download the right file for your system from the [GitHub Releases page](https://github.com/enapt/SwarmLLM/releases/latest):

| Your Computer | File Name |
|---|---|
| **Windows** (most PCs) | `swarmllm-windows-x86_64.zip` |
| **Mac** (M1/M2/M3/M4) | Coming soon |
| **Mac** (older Intel) | Coming soon |
| **Linux** (most distros) | `swarmllm-linux-x86_64.tar.gz` |
| **Linux** (NVIDIA GPU) | `swarmllm-linux-x86_64-cuda.tar.gz` |

> **Not sure which Mac?** Apple menu > "About This Mac." If it says "Apple M1" (or M2/M3/etc.), pick Apple Silicon. If it says "Intel," pick Intel.

## Install & Run

### Windows

1. Right-click `swarmllm-windows-x86_64.zip` and choose **Extract All...**
2. Double-click `swarmllm.exe` in the extracted folder.
3. If SmartScreen warns you, click **More info** > **Run anyway**.

Or from PowerShell:
```powershell
cd Downloads\swarmllm-windows-x86_64
.\swarmllm.exe run
```

### macOS

> **Note:** Pre-built macOS binaries are not yet available. Build from source instead (see below).

If macOS blocks a locally-built binary: System Settings > Privacy & Security > click **Open Anyway** next to SwarmLLM.

### Linux

```bash
cd ~/Downloads
tar xzf swarmllm-linux-x86_64.tar.gz
cd swarmllm-linux-x86_64
chmod +x swarmllm
./swarmllm run
```

### Cargo Install

Requires Rust 1.80+:

```bash
cargo install --git https://github.com/enapt/SwarmLLM.git --tag v0.1.0-alpha.1
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

For Apple Silicon (Metal):
```bash
cargo build --release --features metal
```

## Open the Dashboard

Once running, open **[http://localhost:8800](http://localhost:8800)** in your browser. The setup wizard will walk you through initial configuration.
