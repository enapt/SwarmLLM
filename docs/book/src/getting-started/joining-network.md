# Joining the Network

SwarmLLM works standalone, but connecting to peers unlocks distributed inference for larger models.

## Automatic Discovery

SwarmLLM finds peers automatically:

- **Same network (LAN):** mDNS discovers peers on the same Wi-Fi/LAN in seconds.
- **Returning users:** Previously-seen peers are remembered and reconnected on startup.
- **Peer exchange:** Connected peers share their peer lists with you.

## Invite Codes (Easiest)

1. In the Dashboard, click **"Share Network Code"**
2. Copy the encrypted code and share it with a friend
3. They paste it into the **"Join Network"** field and click **Join**
4. Both nodes connect immediately and start discovering the wider network

> Invite codes are encrypted (ChaCha20Poly1305) — your IP address is not visible in the code itself. Anyone with the full code can decode it, but the IP can't be extracted by casual inspection.

## Manual Bootstrap

```bash
./swarmllm run --bootstrap "/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW..."
```

Or in your config file:
```toml
[network]
bootstrap_peers = ["/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW..."]
```

## Private Networks

`gossip_network_id` puts a group of computers on their own announcement
channels, with their own key, so they discover each other's models separately
from the public network:

```toml
[network]
gossip_network_id = "my-private-network"
```

**On its own this is not an isolation boundary, and it is important not to
treat it as one.** It separates announcements. It does not stop your computer
sending work to a computer outside the group: which machines hold which model
parts is also learned through the shared peer-to-peer directory, which every
SwarmLLM node takes part in regardless of this setting. Measured on a test
group whose very first request was answered by a public node.

**To actually keep work inside a set of machines, use a pool and turn on
private mode:**

```toml
[pool]
private_mode_allow_lan = false   # default is true, and "LAN" includes any
                                 # other node on the same network
```

Then link the machines into a pool and switch private mode on. That is an
explicit list of computers, checked before any work is handed out, and it is
the only thing that decides where your prompts go. Use both together for a
private cluster: `gossip_network_id` to keep the announcements separate, pool
private mode to keep the work in.

Note `private_mode_allow_lan` is read at startup, so it belongs in the config
file before you start the node rather than being changed while it runs.

## Firewall & internet reachability

SwarmLLM needs **TCP port 8810** (P2P primary transport) and optionally **UDP port 8800** (QUIC) open. On the same LAN, mDNS handles everything — no ports to open. To be reachable **across the internet** you need one of:

- **UPnP** (on by default) — opens the port on a cooperative home router automatically.
- **Manual port-forward** (TCP 8810 + UDP 8800 to your machine) plus `external_address` in config.
- **A relay/anchor node** — reach the swarm through a publicly-reachable node, even behind CGNAT.

If your invite code says *"only works on your local network,"* your node isn't internet-reachable yet. See **[docs/NETWORKING.md](https://github.com/enapt/SwarmLLM/blob/main/docs/NETWORKING.md)** for the full guide — including the CGNAT check and how to run your own anchor node.
