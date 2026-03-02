# Joining the Network

SwarmLLM works standalone, but connecting to peers unlocks distributed inference for larger models.

## Automatic Discovery

SwarmLLM finds peers automatically:

- **Same network (LAN):** mDNS discovers peers on the same Wi-Fi/LAN in seconds.
- **Returning users:** Previously-seen peers are remembered and reconnected on startup.
- **Peer exchange:** Connected peers share their peer lists with you.

## Invite Codes (Easiest)

1. In the Dashboard, find the **"Your Network Code"** section
2. Share the code with a friend
3. They paste it into the **"Join Network"** field and click **Join**
4. Both nodes connect and start discovering the wider network

> The invite panel auto-hides once your node knows 20+ peers.

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

## Firewall

SwarmLLM needs **UDP port 8800** open. If you're behind a router, either:
- Set up port forwarding (UDP 8800 to your machine's local IP)
- Rely on SwarmLLM's built-in relay (works automatically in most cases)
