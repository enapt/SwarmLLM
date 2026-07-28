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
- **Dashboard requires the API key** — admin endpoints need Bearer token auth. The dashboard fetches that key for itself on page load, but only over networks the node trusts (see below)

## Opening the dashboard over Tailscale

Admin endpoints need a Bearer token. The dashboard normally obtains one for
itself on page load, and the daemon decides whether to hand it over based on the
source address of the request:

| Source | Handed a key? |
|---|---|
| Loopback (`127.0.0.1`, `::1`) | Always |
| Tailscale (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`) | Yes, **if this node is itself on a tailnet** |
| Private / LAN address | Only with `api.dashboard_trust_lan = true` |
| Anything else | No — paste the key instead |

So if SwarmLLM runs on a machine that has joined your tailnet, browsing to
`http://100.x.x.x:8800` from another device on the tailnet just works, with no
configuration. Set `api.dashboard_trust_overlay = false` to turn that off — worth
doing if you share the tailnet with people you would not give admin access to.

The IPv4 range is shared CGNAT space that some ISPs also use, so it is not on its
own proof of a tailnet. Trust is only extended across it when this node holds an
overlay address too, which an ISP's CGNAT segment does not give it.

### If you reach the node through a subnet router

**A subnet router masquerades by default.** If Tailscale runs on a Proxmox host
(or any gateway) and advertises the subnet a SwarmLLM container sits on, the
container does not see your device's `100.x` address — it sees the *router's*
private address, which is indistinguishable from any other LAN client. The same
is true of a container port publish or any NAT hop.

Two ways through, in order of preference:

1. **Unlock the dashboard once with the key.** The page will say it wasn't given
   a key and offer a box to paste one. The key is printed when SwarmLLM starts
   and stored in the `api_key` file in its data directory — read it from inside
   the container. It's remembered per node, so this is a one-time step per
   browser.
2. **Turn on "Allow access from my local network"** in Settings → Identity &
   Access (or `api.dashboard_trust_lan = true`). This admits any private address,
   including the router's, and applies immediately without restarting the node.
   Only do this on a network whose devices you trust — it is a weaker boundary
   than the tailnet, which at least authenticates its members.

Alternatively, stop the router masquerading so the original address survives, and
the tailnet rule above applies unchanged:

```bash
tailscale up --advertise-routes=<subnet> --snat-subnet-routes=false
```

That requires the destination to route back over the tailnet, and is Linux-only.

Inference and the OpenAI/Anthropic APIs are unaffected by any of this — they
accept the API key as a Bearer token from any address.

## Troubleshooting

**Dashboard loads but nothing saves (setup wizard's "Start SwarmLLM" appears to
do nothing):** every admin call is returning 401 because the page was never given
a key. The banner at the top of the page names the address the daemon actually
saw for you — which behind a NAT or subnet router is *not* the address in your
browser's address bar — and offers the paste box. See the section above.

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
