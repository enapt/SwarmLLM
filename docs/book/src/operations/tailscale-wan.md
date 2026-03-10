# Tailscale & WAN Access

SwarmLLM works over any IP-routable network, including VPN overlays like [Tailscale](https://tailscale.com), WireGuard, and ZeroTier. This guide covers how to access your node remotely and connect peers across the internet.

## Use Cases

- **Remote access** — Chat with your home GPU from your laptop at a coffee shop
- **Multi-site cluster** — Connect nodes at home and work into one swarm
- **Team deployment** — Share a private swarm across your team without exposing ports to the internet
- **Cloud + local hybrid** — Connect a cloud GPU instance to your local network

## Quick Setup with Tailscale

### 1. Install Tailscale on all machines

```bash
# Linux
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

# macOS
brew install tailscale
tailscale up

# Windows — download from https://tailscale.com/download
```

Each machine gets a stable `100.x.x.x` IP address on the Tailscale network.

### 2. Start SwarmLLM normally

```bash
# On each machine — no special flags needed
./swarmllm run
```

SwarmLLM binds to `0.0.0.0` by default, which includes the Tailscale interface.

### 3. Connect peers via bootstrap

Since mDNS doesn't work across Tailscale (it's link-local only), use one of these methods:

**Option A: Invite code (easiest)**

On Node A, copy the invite code from the dashboard (`http://localhost:8800`). On Node B, paste it into the "Join Network" field. The invite code contains the node's addresses — including the Tailscale IP if it's listening on `0.0.0.0`.

**Option B: Bootstrap peers in config**

```toml
# ~/.local/share/swarmllm/config.toml on Node B
[network]
bootstrap_peers = [
  "/ip4/100.64.0.5/tcp/8810",    # Node A's Tailscale IP
]
```

**Option C: CLI flag**

```bash
./swarmllm run --bootstrap /ip4/100.64.0.5/tcp/8810
```

### 4. Access the dashboard remotely

Once connected via Tailscale, open the dashboard from any machine:

```
http://100.64.0.5:8800
```

The API is also accessible at that address:

```bash
curl http://100.64.0.5:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Recommended Config for WAN / Tailscale

```toml
[network]
enable_mdns = false           # mDNS is LAN-only, won't work through Tailscale
enable_autonat = false        # Tailscale handles NAT, disable noisy probes
enable_dcutr = false          # Hole punching unnecessary on Tailscale
enable_relay = true           # Keep as fallback for robustness
enable_quic = true            # QUIC works well on Tailscale (low-latency UDP)
bootstrap_peers = [
  "/ip4/100.64.0.5/tcp/8810", # Replace with your peer's Tailscale IP
]
```

For higher latency links (cross-continent), you may also want:

```toml
[inference]
tp_max_latency_ms = 50        # Relax tensor parallelism latency threshold (default: 10ms)
```

## Binding to a Specific Interface

If you only want SwarmLLM accessible via Tailscale (not the local network):

```toml
[network]
listen_address = "100.64.0.5"  # Bind only to Tailscale interface
```

Or bind to localhost only and use Tailscale's [Funnel](https://tailscale.com/kb/1223/funnel) or port forwarding:

```toml
[network]
listen_address = "127.0.0.1"
```

## WireGuard / ZeroTier / Other VPNs

The same approach works with any VPN overlay:

1. Install the VPN on all machines
2. Start SwarmLLM with default config (`listen_address = "0.0.0.0"`)
3. Use the VPN IP as a bootstrap peer address
4. Disable mDNS if peers aren't on the same physical LAN

## Security Notes

- **API key still required** — remote access to inference endpoints requires Bearer token auth, even over Tailscale
- **E2E encryption is independent of VPN** — SwarmLLM encrypts all P2P traffic with X25519 + ChaCha20-Poly1305 regardless of whether you use a VPN. The VPN adds a second layer of encryption at the network level
- **Dashboard is not auth-protected** — the admin dashboard at `/admin` doesn't require authentication. If exposing to untrusted networks, use Tailscale ACLs to restrict access or bind to `127.0.0.1` and use SSH tunneling

## Troubleshooting

**Peers don't connect:**
- Verify Tailscale is running: `tailscale status`
- Check that port 8810 (TCP) and 8800 (UDP/QUIC) are reachable: `tailscale ping 100.64.0.5`
- Try with `--bootstrap /ip4/<TAILSCALE_IP>/tcp/8810` explicitly
- Check logs with `-vv` for connection errors

**Slow inference across WAN:**
- Pipeline parallelism (splitting layers across nodes) works best on low-latency links (<50ms)
- Tensor parallelism requires LAN-like latency (<10ms) — increase `tp_max_latency_ms` or let SwarmLLM use pipeline mode instead
- Consider having each site run its own models for local inference, with the swarm as fallback

**Stale peer cache after IP change:**
- If your Tailscale IP changes, old cached addresses will fail. Delete the database to clear the cache:
  ```bash
  rm ~/.local/share/swarmllm/db.redb
  ```
