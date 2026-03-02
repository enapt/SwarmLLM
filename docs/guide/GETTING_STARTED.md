# Getting Started with SwarmLLM

## What is SwarmLLM?

SwarmLLM lets you run AI language models (like ChatGPT-style assistants) on your own computer — and team up with other people's computers to run even bigger models together. Think of it like a group project: everyone chips in some computing power, and everyone benefits from faster, more capable AI. It's free, open-source, and your conversations stay private.

---

## Step 1: Download SwarmLLM

Download the right file for your computer from the [GitHub Releases page](https://github.com/swarmllm/swarmllm/releases/latest):

| Your Computer | Download This | File Name |
|---|---|---|
| **Windows** (most PCs) | Windows (64-bit) | `swarmllm-windows-x86_64.zip` |
| **Mac** (M1, M2, M3, M4) | macOS Apple Silicon | `swarmllm-macos-aarch64.tar.gz` |
| **Mac** (older Intel) | macOS Intel | `swarmllm-macos-x86_64.tar.gz` |
| **Linux** (most distros) | Linux (64-bit) | `swarmllm-linux-x86_64.tar.gz` |
| **Docker** | See Docker section below | — |

> **Not sure which Mac you have?** Click the Apple menu in the top-left corner, then "About This Mac." If it says "Apple M1" (or M2, M3, etc.), pick Apple Silicon. If it says "Intel," pick Intel.

---

## Step 2: Install & Run

### Windows

1. **Extract the zip:** Right-click `swarmllm-windows-x86_64.zip` and choose **Extract All...**
2. **Run it:** Double-click `swarmllm.exe` in the extracted folder.
   - If Windows SmartScreen shows a warning, click **More info**, then **Run anyway**. (SwarmLLM is open-source and safe.)
3. Your web browser should open automatically. If it doesn't, go to Step 3.

**Or, from PowerShell:**
```powershell
cd Downloads\swarmllm-windows-x86_64
.\swarmllm.exe run
```

### macOS

1. **Extract the archive:** Double-click `swarmllm-macos-aarch64.tar.gz` (or the Intel version) in Finder. This creates a `swarmllm` folder.
2. **Open Terminal:** Press `Cmd + Space`, type `Terminal`, press Enter.
3. **Navigate to the folder and run:**
   ```bash
   cd ~/Downloads/swarmllm-macos-aarch64
   ./swarmllm run
   ```
4. **If macOS blocks it:** You'll see a message saying the app "cannot be verified." Go to **System Settings > Privacy & Security**, scroll down, and click **Open Anyway** next to the SwarmLLM entry. Then run the command again.

### Linux

1. **Extract and run:**
   ```bash
   cd ~/Downloads
   tar xzf swarmllm-linux-x86_64.tar.gz
   cd swarmllm-linux-x86_64
   chmod +x swarmllm
   ./swarmllm run
   ```

### Docker

Run SwarmLLM in one command (no installation needed):

```bash
docker run -p 8800:8800 -v swarmllm-data:/root/.local/share/swarmllm ghcr.io/swarmllm/swarmllm:latest
```

This stores your data in a Docker volume called `swarmllm-data` so it persists between runs.

---

## Step 3: Open the Dashboard

Once SwarmLLM is running, open your web browser and go to:

**[http://localhost:8800](http://localhost:8800)**

> Your browser should open automatically on first run. If it doesn't, just type the address above into your browser's address bar.

### What you'll see: the Setup Wizard

On your first run, SwarmLLM walks you through a quick 4-step setup:

1. **Hardware Detection** — SwarmLLM scans your computer and shows what it found (GPU, RAM, disk space). You don't need to do anything here, just review and continue.
2. **Resource Contribution** — Choose how much of your computer's resources SwarmLLM can use. "Minimal" is a safe default if you're unsure. You can also enable "Auto-manage shards" to let SwarmLLM automatically download parts of popular AI models.
3. **Model Selection** — Pick which AI models to download. Start small — you can always add more later.
4. **Review & Start** — Confirm your choices and you're done!

After the wizard, you'll see the **Dashboard** with stats about your node: connected peers, credit balance, and which models you have.

---

## Step 4: Get Your First Model

You need at least one AI model before you can chat. Here's how to get one:

1. In the dashboard, click the **Browse HuggingFace** button (in the Models section) or click the **+** button next to the model dropdown in the top-right.
2. The **Model Browser** opens. Type a model name in the search box — try `TinyLlama` for a small, fast model that works on any computer.
3. Browse the results. You'll see model names, sizes, and quantization types (smaller numbers like Q4 = smaller file, less memory needed).
4. Click **Download** on the model you want.
5. Wait for the download to finish. You can watch the progress in the Models section of the dashboard — shards (pieces of the model) will turn green as they complete.

### Recommended first models by hardware

| Your Hardware | Recommended Model | Size |
|---|---|---|
| Any computer (testing) | TinyLlama 1.1B Q4_K_M | ~700 MB |
| 8 GB RAM, no GPU | Qwen2.5-3B Q4_K_M | ~2 GB |
| 8 GB VRAM (GPU) | Qwen2.5-7B Q4_K_M | ~4.5 GB |
| 16+ GB VRAM (GPU) | Llama-3-13B Q4_K_M | ~7 GB |

> **What are shards?** Large AI models are split into smaller pieces called "shards" so they can be shared across the network. SwarmLLM handles this automatically — you just pick a model and download.

---

## Step 5: Start Chatting

1. Click the **Chat** tab in the top navigation bar.
2. Select your downloaded model from the dropdown in the top-right corner.
3. Type a message in the text box at the bottom and press **Enter** (or click **Send**).
4. Watch the AI respond! Responses stream in word-by-word.

You can create multiple chat sessions using the **+** button in the left sidebar.

---

## Step 6: Join a Network (Optional)

SwarmLLM works fine on its own, but it's more powerful when connected to other nodes. When you connect to friends, you can:

- **Share models:** If your friend has a model you don't, you can use it through the network.
- **Run bigger models:** Multiple computers can team up to run models that are too large for one machine.
- **Earn credits:** You earn credits by helping process requests for others, which gives you priority when you need help.

### Automatic discovery

SwarmLLM finds peers automatically in several ways:

- **Same network (LAN):** If you and a friend are on the same Wi-Fi or local network, your nodes discover each other automatically via mDNS — no configuration needed.
- **Returning users:** SwarmLLM remembers peers from previous sessions and reconnects on startup.
- **Peer exchange:** Once connected to any peer, your node automatically discovers more through them.

### Using invite codes (easiest way to connect)

1. In the Dashboard, look for the **"Your Network Code"** section. Copy the code shown there.
2. Share it with your friend (text, email, chat — whatever works).
3. Your friend pastes the code into the **"Join Network"** field on their Dashboard and clicks **Join**.
4. Done! Both nodes connect and start discovering the wider network.

> The invite code panel automatically hides itself once your node knows 20+ peers — at that point, the network is self-sustaining.

### Manual bootstrap (advanced)

You can also connect using raw multiaddr strings:

**Command line:**
```bash
./swarmllm run --bootstrap "/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW..."
```

**Config file:**
```toml
[network]
bootstrap_peers = ["/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW..."]
```

After connecting, your Dashboard should show "Connected Peers: 1" (or more). The network grows automatically from there.

---

## What's Next?

- **[Configuration Guide](CONFIGURATION.md)** — Customize your port, nickname, VRAM limits, and more.
- **[Troubleshooting](TROUBLESHOOTING.md)** — Solutions for common issues (can't connect, GPU not detected, etc.).

### Useful keyboard shortcuts

| Action | Shortcut |
|---|---|
| Send message | `Enter` |
| New line in message | `Shift + Enter` |

### Quick commands

```bash
./swarmllm run                  # Start the node (default port 8800)
./swarmllm run -p 9000          # Start on a different port
./swarmllm run -v               # Start with more detailed logging
./swarmllm status               # Check if the node is running
./swarmllm version              # Show version number
./swarmllm chat                 # Interactive chat in the terminal
./swarmllm bench                # Run inference benchmarks (tokens/sec)
./swarmllm peers                # List connected peers and their status
```

---

## Using the Python SDK

If you prefer Python, install the SwarmLLM client SDK:

```bash
pip install swarmllm-client
```

Then use it like the OpenAI SDK:

```python
from swarmllm_client import SwarmLLMClient

client = SwarmLLMClient(base_url="http://localhost:8800", api_key="YOUR_API_KEY")
response = client.chat.completions.create(
    model="your-model-name",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

The SDK supports auto-discovery of local nodes, streaming, embeddings, and tool calls. See the `python/` directory in the repository for full documentation.

---

## Monitoring Stack (Optional)

For production deployments, SwarmLLM ships a ready-to-use Grafana + Prometheus monitoring stack:

```bash
cd monitoring/
docker compose up -d
```

This starts Prometheus (scrapes `/metrics` from your node) and Grafana (pre-configured dashboards for inference latency, peer count, credit balance, VRAM usage, and more). Open Grafana at `http://localhost:3000`.

See `monitoring/README.md` for configuration details.
