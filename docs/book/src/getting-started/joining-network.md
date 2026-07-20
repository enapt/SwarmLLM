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

To run a private cluster that doesn't mix with the public network:

```toml
[network]
gossip_network_id = "my-private-network"
```

Only nodes with the same `gossip_network_id` can communicate.

## Firewall & internet reachability

SwarmLLM needs **TCP port 8810** (P2P primary transport) and optionally **UDP port 8800** (QUIC) open. On the same LAN, mDNS handles everything — no ports to open. To be reachable **across the internet** you need one of:

- **UPnP** (on by default) — opens the port on a cooperative home router automatically.
- **Manual port-forward** (TCP 8810 + UDP 8800 to your machine) plus `external_address` in config.
- **A relay/anchor node** — reach the swarm through a publicly-reachable node, even behind CGNAT.

If your invite code says *"only works on your local network,"* your node isn't internet-reachable yet. See **[docs/NETWORKING.md](https://github.com/enapt/SwarmLLM/blob/main/docs/NETWORKING.md)** for the full guide — including the CGNAT check and how to run your own anchor node.
