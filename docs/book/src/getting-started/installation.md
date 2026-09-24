# Installation

## Download

Download the right file for your system from the [GitHub Releases page](https://github.com/enapt/SwarmLLM/releases/latest):

| Your Computer | File Name |
|---|---|
| **Windows** (NVIDIA, AMD or Intel graphics card) | `swarmllm-windows-x86_64-gpu.zip` |
| **Windows** (no graphics card, or not sure) | `swarmllm-windows-x86_64-cpu.zip` |
| **Mac** (Apple chip — M1 or newer) | `swarmllm-macos-aarch64.tar.gz` (runs on the processor; no graphics acceleration yet) |
| **Mac** (Intel chip) | Not supported yet — build from source (best-effort) |
| **Linux** (most distros) | `swarmllm-linux-x86_64.tar.gz` |
| **Linux** (NVIDIA GPU) | `swarmllm-linux-x86_64-cuda.tar.gz` |
| **Linux** (processor from before 2013) | `swarmllm-linux-x86_64-baseline.tar.gz` |
| **Windows** (processor from before 2013) | `swarmllm-windows-x86_64-baseline.zip` |

> **It stops immediately with `Illegal instruction`?** Download the
> `-baseline` build. The ordinary Linux and Windows builds are made for
> processors from 2013 onwards (they use AVX2 instructions), which makes them
> roughly 3x faster at running models on the processor. On an older machine
> they stop on the very first instruction, before printing anything — and
> because nothing is wrong with the download, re-downloading or checking
> permissions changes nothing. On Linux, `grep -c avx2 /proc/cpuinfo` printing
> `0` means you want the baseline build.

> **Not sure which Mac?** Apple menu > "About This Mac." If it says "Apple M1" (or M2/M3/etc.), use the Mac download. If it says "Intel," there is no download for your Mac yet.

> **Which NVIDIA cards get GPU acceleration?** RTX 30-series and newer (also
> RTX 40, RTX 50, and the A/H data-centre cards). The RTX 20-series, GTX
> 16-series and anything older are below the requirement of the FlashAttention
> kernels SwarmLLM ships, which is what makes attention fast.
>
> **An older card is not a problem** — nothing breaks and there is nothing to
> configure. SwarmLLM checks the card when it starts, tells you in the log and
> on the dashboard that it is using the processor instead, and carries on. On
> Windows, running a model locally goes through Vulkan and works on any GPU
> regardless; the CUDA requirement applies to inference split across several
> machines.
>
> To check your card: `nvidia-smi --query-gpu=name,compute_cap --format=csv`.
> A number of 8.0 or higher gets GPU acceleration.

## Check the download (optional, one command)

The single-file downloads — `swarmllm-linux-x86_64`,
`swarmllm-windows-x86_64-gpu.exe` and the like, plus the `.deb` and `.rpm` —
each have a `.sha256` file beside them on the release page. (The `.zip` and
`.tar.gz` archives don't.) Put both files in the same folder and run one
command to catch a damaged or incomplete download:

```bash
sha256sum -c swarmllm-linux-x86_64.sha256            # Linux
shasum -a 256 -c swarmllm-macos-aarch64.sha256       # Mac
```

Those `.sha256` files are also signed, and the built-in updater refuses an
update whose signature it cannot verify.

## Install & Run

### Windows

1. Download `swarmllm-windows-x86_64-gpu.zip` (any NVIDIA, AMD or Intel graphics card — it bundles the NVIDIA libraries, so no CUDA Toolkit is needed) or `swarmllm-windows-x86_64-cpu.zip` (works on every PC).
2. Right-click it and choose **Extract All** — running it from inside the zip view does not work.
3. Double-click `swarmllm.exe` in the extracted folder. SmartScreen shows *"Windows protected your PC"* because the program is not code-signed yet: click **More info** > **Run anyway**.
4. A console window opens: that is SwarmLLM running. Keep it open (minimising is fine) — closing it stops SwarmLLM. The dashboard opens in your browser.

> **There is no `SwarmLLM-Setup.exe` at the moment.** The installer that
> bundled both variants was dropped from the release build on 2026-04-22 and
> has not been published since; earlier versions of this page still named it.

From PowerShell on a raw binary:
```powershell
cd Downloads\swarmllm-windows-x86_64-gpu
.\swarmllm.exe run
```

### macOS

Double-click the download in Finder to unpack it into a folder, keep that
folder somewhere in your home folder (Documents or Downloads is fine), and
double-click `swarmllm` inside it. A Terminal window opens — that is
SwarmLLM running, so leave it open. Or from Terminal (the archive has no
folder of its own, so unpack it into one):

```bash
mkdir -p ~/swarmllm
tar xzf ~/Downloads/swarmllm-macos-aarch64.tar.gz -C ~/swarmllm
cd ~/swarmllm
./swarmllm run
```

> **Where you put it matters on a Mac.** Keep SwarmLLM in a folder you own — anywhere under your home folder, as in the commands above. Folders like `/Applications` need administrator rights, and SwarmLLM cannot then replace its own binary, so it will tell you an update is available and decline to install it. Nothing else about it changes; move the file and updates work by themselves.

> **First launch: "cannot be opened because it is from an unidentified developer."** The binaries are not yet signed with an Apple developer certificate, so Gatekeeper stops the first run of anything downloaded **in a browser**. Either open System Settings > Privacy & Security and click **Open Anyway**, or remove the quarantine flag the browser added:
>
> ```bash
> xattr -d com.apple.quarantine swarmllm
> ```
>
> Downloading with `curl` avoids it entirely — the quarantine flag is set by the browser, not by macOS in general — and it is a one-time thing per download, not something that recurs on every update (confirmed on a Mac mini M4, 2026-09-03).

### Linux

The archive has no folder of its own, so unpack it into one (use
`swarmllm-linux-x86_64-cuda.tar.gz` for an NVIDIA RTX 30-series card or newer):

```bash
mkdir -p ~/swarmllm
tar xzf ~/Downloads/swarmllm-linux-x86_64.tar.gz -C ~/swarmllm
cd ~/swarmllm
./swarmllm run
```

### Linux packages (.deb / .rpm)

```bash
sudo dpkg -i swarmllm_*_amd64.deb                      # Debian / Ubuntu
sudo systemctl enable --now swarmllm                   # start it as a background service
sudo rpm -i swarmllm_*.x86_64.rpm                      # Fedora / RHEL
```

The .deb sets SwarmLLM up as a background service that keeps its data in
`/var/lib/swarmllm` (the access token is in `/var/lib/swarmllm/api_key`).
Programs installed from a package **do not update themselves** — install the
next release's package to update. Homebrew and AUR packages are not published
yet.

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
| `ghcr.io/enapt/swarmllm:<version>` | A pinned release, e.g. `0.3.200-alpha` (CPU) |
| `ghcr.io/enapt/swarmllm:<version>-cuda` | A pinned release (GPU) |

An image is published when its release is — after the release has been checked
and signed, the same moment the built-in updater can see it. Don't pin
`0.3.199-alpha`: that release was withdrawn, and before this rule its image was
published anyway (as were `0.3.202-alpha` and `0.3.203-alpha`, which were never
released).

Data is persisted in Docker volumes. Model shards are stored in the `swarmllm-models` volume (or bind-mount a host directory via `SWARMLLM_MODELS_DIR` in `.env`).

View logs with `docker compose logs -f`. The API key is printed on first startup.

### Cargo Install

Requires Rust 1.90+:

```bash
cargo install --git https://github.com/enapt/SwarmLLM.git   # add --tag vX.Y.Z-alpha to pin a release
swarmllm run
```

### Building from Source

```bash
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM
cargo build --release
./target/release/swarmllm run
```

For NVIDIA GPU support, build what the release builds (llama.cpp + candle on
CUDA + FlashAttention; it needs the CUDA toolkit and compiles for a long time):
```bash
cargo build --release --features cuda
```
Every feature flag is listed in `CONTRIBUTING.md`.

For Apple Silicon: the default build runs on CPU. A Metal-accelerated
build is on the roadmap but not yet implemented (no `metal` Cargo
feature exists yet); until then, use the default `cargo build --release`.

## Open the Dashboard

Once running, open **[http://localhost:8800](http://localhost:8800)** in your browser. The setup wizard will walk you through initial configuration.
