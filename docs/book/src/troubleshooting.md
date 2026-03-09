# Troubleshooting

## Can't Connect to Peers

**Check the bootstrap address format:**
```
/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW...
```

**Firewall:** SwarmLLM needs **TCP port 8810** (P2P) and optionally **UDP port 8800** (QUIC) open.
- **Linux:** `sudo ufw allow 8810/tcp && sudo ufw allow 8800/udp`
- **Windows:** Windows Defender Firewall > Inbound Rules > New > Port > TCP 8810 + UDP 8800
- **macOS:** System Settings > Network > Firewall > allow SwarmLLM

**Same LAN?** Use local IP (e.g., `192.168.1.x`). LAN peers should be found automatically via mDNS.

## Model Download Stuck

1. Check disk space — a 7B model needs ~4-5 GB free
2. Verify internet access to `https://huggingface.co`
3. Cancel and retry from the Dashboard
4. Start with `-v` for verbose logs: `./swarmllm run -v`
5. Try a smaller model first (TinyLlama, ~700 MB)

## GPU Not Detected

1. Verify GPU works: `nvidia-smi`
2. Install NVIDIA drivers if needed
3. Enable GPU offloading: `./swarmllm run --gpu-layers 99`

**WSL2 users:** The CUDA driver comes from your Windows NVIDIA driver. Check that `/usr/lib/wsl/lib/libcuda.so.1` exists and add to your `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Port Already in Use

```bash
./swarmllm run --port 9000    # Use a different port
lsof -i :8800                 # Find what's using 8800
./swarmllm status             # Check if another instance is running
```

## Slow First Request

If the first inference request to a model takes noticeably longer than subsequent ones, this is expected. SwarmLLM uses **on-demand model loading** — models whose shards are on disk but not loaded into VRAM are loaded when first requested. If VRAM is full, an LRU eviction occurs first. Subsequent requests to the same model will be fast.

## Slow Inference

1. **GPU vs CPU:** CPU is 5-20x slower. Check Dashboard for GPU status.
2. **Model too large:** Use Q4 quantization, match model size to VRAM.
3. **Enable batching:** Set `max_batch_size = 4` in config.

## Database Corrupted

```bash
# Back up first
cp -r ~/.local/share/swarmllm ~/.local/share/swarmllm-backup
# Delete database (models and config are preserved)
rm -rf ~/.local/share/swarmllm/db
# Restart
./swarmllm run
```

## Still Stuck?

- Check [GitHub Issues](https://github.com/enapt/SwarmLLM/issues)
- Open a new issue with: OS, hardware, `./swarmllm version`, and logs from `-vv`
