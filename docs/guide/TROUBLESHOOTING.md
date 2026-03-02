# Troubleshooting

Having trouble? You're probably not the first — here are solutions to the most common issues.

---

## "I can't connect to any peers"

**Symptoms:** Dashboard shows "0 Connected Peers" even after adding a bootstrap address.

**Things to check:**

1. **Is the bootstrap address correct?** It should look exactly like this:
   ```
   /ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW...
   ```
   Make sure there are no extra spaces or missing parts. The whole thing is one long string.

2. **Is the other node actually running?** Ask your friend to check that their SwarmLLM is running and that their Dashboard shows a green status dot.

3. **Are you both on the same port?** By default, SwarmLLM uses port **8800**. If either of you changed the port, make sure the bootstrap address uses the correct port number.

4. **Firewall blocking connections?** SwarmLLM needs **UDP port 8800** open for peer-to-peer connections.

   - **Windows:** Open Windows Defender Firewall > Advanced Settings > Inbound Rules > New Rule > Port > UDP > 8800 > Allow.
   - **macOS:** System Settings > Network > Firewall > allow SwarmLLM.
   - **Linux:**
     ```bash
     sudo ufw allow 8800/udp
     ```

5. **Behind a router?** If the other node is on a different network (not the same Wi-Fi), you may need to set up **port forwarding** on your router. Forward UDP port 8800 to your computer's local IP address. Check your router's manual for how to do this.

   > SwarmLLM has built-in NAT traversal (relay) that often works without port forwarding, but it depends on your network setup.

6. **Same local network?** If you're on the same Wi-Fi, use the local IP (like `192.168.1.x`) instead of a public IP. You can find your local IP with:
   - **Windows:** Open PowerShell and run `ipconfig`
   - **macOS/Linux:** Open Terminal and run `ifconfig` or `ip addr`

---

## "Model download is stuck"

**Symptoms:** You clicked Download but the progress bar isn't moving, or shards aren't turning green.

**Things to try:**

1. **Check your disk space.** AI models are large — make sure you have enough free space. A 7B model needs about 4–5 GB free.
   - **Windows:** Open File Explorer, right-click your drive, Properties.
   - **macOS:** Apple menu > About This Mac > Storage.
   - **Linux:** Run `df -h` in Terminal.

2. **Check your internet connection.** Models are downloaded from HuggingFace (a model hosting service). Make sure you can access https://huggingface.co in your browser.

3. **Cancel and retry.** In the Dashboard, find the model in the Models section. If there's a cancel option, use it, then try downloading again.

4. **Check the logs.** Start SwarmLLM with verbose logging to see what's happening:
   ```bash
   ./swarmllm run -v
   ```
   Look for error messages related to downloads or network timeouts.

5. **Try a smaller model first.** If a large model keeps failing, try downloading TinyLlama (about 700 MB) to make sure downloads work at all.

---

## "CUDA/GPU not detected"

**Symptoms:** The Dashboard shows "CPU" under GPU, or inference is very slow even though you have an NVIDIA GPU.

### General steps

1. **Check that your GPU is recognized by your system:**
   ```bash
   nvidia-smi
   ```
   This should show your GPU name, temperature, and memory. If it says "command not found," you need to install NVIDIA drivers.

2. **Install NVIDIA drivers:**
   - **Windows:** Download from [nvidia.com/drivers](https://www.nvidia.com/drivers/)
   - **Linux:**
     ```bash
     sudo apt install nvidia-driver-535    # Ubuntu/Debian
     ```
   - **macOS:** NVIDIA GPUs are not supported on modern macOS. Use CPU mode.

3. **Enable GPU layers.** By default, SwarmLLM uses CPU only (`gpu_layers = 0`). You need to tell it to use the GPU:
   ```bash
   ./swarmllm run --gpu-layers 99
   ```
   Or in your config file:
   ```toml
   [inference]
   gpu_layers = 99    # Offload all layers to GPU
   ```

### WSL2 (Windows Subsystem for Linux) specific

If you're running SwarmLLM in WSL2:

1. **Make sure your Windows NVIDIA driver is up to date.** The WSL2 CUDA driver comes from your Windows driver — you do NOT install a separate Linux driver in WSL2.

2. **Check that the CUDA library is accessible:**
   ```bash
   ls /usr/lib/wsl/lib/libcuda.so.1
   ```
   If this file doesn't exist, update your Windows NVIDIA driver.

3. **Set the library path** (add this to your `~/.bashrc`):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
   ```
   Then restart your terminal or run `source ~/.bashrc`.

---

## "Port 8800 is already in use"

**Symptoms:** SwarmLLM fails to start with an error about the port being in use.

**Solutions:**

1. **Use a different port:**
   ```bash
   ./swarmllm run --port 9000
   ```
   Then open `http://localhost:9000` in your browser instead.

2. **Find what's using port 8800:**
   - **Windows (PowerShell):**
     ```powershell
     netstat -ano | findstr :8800
     ```
   - **macOS/Linux:**
     ```bash
     lsof -i :8800
     ```

3. **Is another SwarmLLM instance running?** Only one instance can run per port. Check with:
   ```bash
   ./swarmllm status
   ```

---

## "My inference is slow"

**Symptoms:** Responses take a long time (more than a few seconds for short replies).

**Things to check:**

1. **Are you using GPU or CPU?** CPU inference is 5–20x slower than GPU. Check the Dashboard — if it shows "CPU" under GPU, see the GPU troubleshooting section above.

2. **Is the model too big for your hardware?** If the model uses more memory than you have, it falls back to CPU for the overflow. Recommendations:
   - **4 GB VRAM:** Use 3B models (Q4 quantization)
   - **8 GB VRAM:** Use 7B models (Q4 quantization)
   - **16 GB VRAM:** Use 13B models (Q4 quantization)
   - **24 GB+ VRAM:** Use 30B+ models

3. **Lower the model size or quantization.** Q4 models are faster (and smaller) than Q8. Try downloading a Q4_K_M version if you're using a larger quantization.

4. **Check concurrent requests.** If many people are using your node at once, responses slow down. You can limit this:
   ```toml
   [inference]
   max_concurrent_requests = 5
   ```

5. **Enable batching** for better throughput when handling multiple requests:
   ```toml
   [inference]
   max_batch_size = 4
   batch_timeout_ms = 50
   ```

---

## "How do credits work?"

Credits are SwarmLLM's way of keeping things fair. Here's the simple version:

- **You earn credits** when your node helps other people — specifically, when it processes model layers for someone else's request. You earn **10 credits per layer processed**.
- **You spend credits** when you send a chat request that uses other people's nodes. You spend **8 credits per token generated**.
- **Your tier** is based on your credit balance compared to others on the network:
  - **Bronze** — Negative balance (you've used more than you've contributed)
  - **Silver** — Positive balance
  - **Gold** — Top 30% of contributors
  - **Platinum** — Top 10% of contributors
- **Higher tiers get priority.** When the network is busy, Gold and Platinum users' requests are processed first.
- **You can always use your own models locally** without spending credits — credits only apply to distributed (multi-node) inference.

> **Tip:** Leave SwarmLLM running with auto-manage enabled. Your node will automatically download and serve popular model shards, earning you credits passively.

---

## "How do I update SwarmLLM?"

SwarmLLM can update itself automatically. By default, it checks for stable updates every 6 hours.

**To update manually:**

1. Download the latest version from the [GitHub Releases page](https://github.com/swarmllm/swarmllm/releases/latest).
2. Stop the running SwarmLLM (click the power button in the Dashboard, or press `Ctrl+C` in the terminal).
3. Replace the old `swarmllm` binary with the new one.
4. Start it again with `./swarmllm run`.

Your settings, models, and data are preserved — they live in the data directory, not the binary.

**To control auto-updates:**
```toml
[updates]
auto_update = "stable"    # "disabled" to turn off, "stable" for stable releases, "all" for canary
check_interval_hours = 6
auto_restart = true       # Automatically restart after updating
```

---

## "Can I run multiple models?"

Yes! SwarmLLM can host multiple models at the same time.

- **With auto-manage enabled** (the default), SwarmLLM automatically decides which models to download and serve based on what's popular and what fits in your disk/VRAM budget.
- **Manual downloads:** Use the HuggingFace browser in the Dashboard to download specific models.
- **Switching models:** Use the dropdown in the top-right corner of the Dashboard to switch which model you're chatting with.

The number of models you can run depends on your disk space and VRAM. Models are loaded on demand — only the active model uses VRAM.

---

## "Is my data private?"

**Short answer:** Yes. Your conversations stay on your computer.

**In detail:**

- **Chat messages are never stored on other people's nodes.** When you use distributed inference (multiple nodes working together), only the mathematical data (tensor activations) needed for computation is sent over the network — not your actual text.
- **All network communication is encrypted.** SwarmLLM uses end-to-end encryption (X25519 key exchange + ChaCha20-Poly1305 cipher) for all peer-to-peer data.
- **Your node identity is a cryptographic key pair** (Ed25519). It's generated on your machine and the private key never leaves your computer.
- **What IS shared with the network:**
  - Your node ID (a public key, like a username)
  - Which model shards you're hosting (so others know where to find model pieces)
  - Your region, if you set one (for the network map — this is voluntary)
  - Your nickname, if you set one (also voluntary)
  - Your credit balance and tier (for fairness)

---

## "SwarmLLM crashed / won't start"

1. **Check the logs.** Run with verbose logging:
   ```bash
   ./swarmllm run -vv
   ```
   The error message usually tells you exactly what went wrong.

2. **Database corrupted?** If you see errors about redb or the database, you can reset it:
   ```bash
   # Back up first, just in case:
   cp -r ~/.local/share/swarmllm ~/.local/share/swarmllm-backup
   # Then delete the database:
   rm -f ~/.local/share/swarmllm/db.redb
   ```
   SwarmLLM will recreate the database on next start. Your models and config are preserved.

   **Migrating from sled:** If you are upgrading from an older version that used sled (the `db/` directory), build with the `migrate-sled` feature flag. On first startup, SwarmLLM will automatically migrate all data from the sled `db/` directory into the new `db.redb` file. After verifying the migration succeeded, you can safely delete the old `db/` directory.

3. **Out of memory?** If SwarmLLM crashes during inference, you may need a smaller model or lower `gpu_layers`.

4. **Permission issues (Linux)?** Make sure the binary is executable:
   ```bash
   chmod +x swarmllm
   ```

---

## Common Error Messages

SwarmLLM provides actionable error messages with hints. Here are the most common ones:

| Error Message | Meaning | Fix |
|---|---|---|
| `model not found: <name>` | The requested model is not downloaded on this node. | Download it from the HuggingFace browser in the Dashboard, or use `POST /api/admin/hf/download-shards`. |
| `no peers hosting shard <N>` | The network has no nodes with the required shard for distributed inference. | Wait for more peers to come online, or download the missing shard yourself. |
| `VRAM insufficient: need <X>MB, available <Y>MB` | The model requires more GPU memory than is available. | Use a smaller quantization (Q4 instead of Q8), reduce `gpu_layers`, or free VRAM by closing other GPU applications. |
| `authentication required` | The API endpoint requires a Bearer token. | Include `-H "Authorization: Bearer YOUR_API_KEY"` in your request. Find your key in Dashboard > Settings. |
| `rate limit exceeded` | Too many requests in a short time window. | Wait a moment and retry. Adjust `max_concurrent_requests` in config if needed. |
| `redb migration needed` | An old sled database was detected but the `migrate-sled` feature is not enabled. | Rebuild with `--features migrate-sled` to enable automatic migration, or delete the old `db/` directory to start fresh. |
| `channel backpressure: <subsystem>` | An internal message channel is full, indicating the subsystem is overloaded. | This is usually transient. If persistent, check Prometheus metrics for the bottleneck subsystem. |
| `tensor decompression failed` | A compressed tensor payload could not be decoded. | Ensure both nodes are running the same SwarmLLM version. Check for network corruption. |

---

## Still stuck?

- **Check GitHub Issues:** [github.com/swarmllm/swarmllm/issues](https://github.com/swarmllm/swarmllm/issues) — someone may have already reported your problem.
- **Open a new issue:** Include your OS, hardware (GPU, RAM), the SwarmLLM version (`./swarmllm version`), and the full error message.
- **Logs help!** Attach logs from running with `-vv` when reporting issues.
