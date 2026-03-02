# Installation

## Download

Download the right file for your system from the [GitHub Releases page](https://github.com/swarmllm/swarmllm/releases/latest):

| Your Computer | File Name |
|---|---|
| **Windows** (most PCs) | `swarmllm-windows-x86_64.zip` |
| **Mac** (M1/M2/M3/M4) | `swarmllm-macos-aarch64.tar.gz` |
| **Mac** (older Intel) | `swarmllm-macos-x86_64.tar.gz` |
| **Linux** (most distros) | `swarmllm-linux-x86_64.tar.gz` |

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

```bash
cd ~/Downloads
tar xzf swarmllm-macos-aarch64.tar.gz
cd swarmllm-macos-aarch64
./swarmllm run
```

If macOS blocks it: System Settings > Privacy & Security > click **Open Anyway** next to SwarmLLM.

### Linux

```bash
cd ~/Downloads
tar xzf swarmllm-linux-x86_64.tar.gz
cd swarmllm-linux-x86_64
chmod +x swarmllm
./swarmllm run
```

### Docker

```bash
docker run -p 8800:8800 -v swarmllm-data:/root/.local/share/swarmllm ghcr.io/swarmllm/swarmllm:latest
```

### Building from Source

Requires Rust 1.75+:

```bash
git clone https://github.com/swarmllm/swarmllm.git
cd swarmllm
cargo build --release
./target/release/swarmllm run
```

For GPU (CUDA) support:
```bash
cargo build --release --features cuda
```

## Open the Dashboard

Once running, open **[http://localhost:8800](http://localhost:8800)** in your browser. The setup wizard will walk you through initial configuration.
