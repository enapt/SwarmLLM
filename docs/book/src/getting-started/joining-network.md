# Joining the Network

You don't have to do anything to join: a new SwarmLLM connects to the public swarm by itself through a built-in starting computer, then finds more computers from there.

## Automatic Discovery

SwarmLLM finds other computers automatically:

- **Public swarm:** on first start SwarmLLM contacts a built-in starting computer and learns about others from it.
- **Same network (LAN):** computers on the same Wi-Fi or home network find each other within seconds.
- **Returning users:** computers you've connected to before are remembered and reconnected on start-up.
- **Peer exchange:** connected computers share their lists of other computers with you.

## Connect to a specific computer

To connect straight to a friend's computer:

1. On the **Dashboard**, find the **Computers** panel and click **Connect a computer**.
2. Under **Your Swarm Address**, click **Copy** and send the `swarm://…` address to your friend.
3. Your friend opens the same panel, pastes it under **Another User's Swarm Address**, and clicks **Connect**.

> The address is scrambled so your IP address isn't readable at a glance, but anyone who has the whole address can decode it — share it only with people you'd give your IP address to.

This connects you to the SwarmLLM network. To link **your own** devices into a private group, use **More → My Devices** instead (see [Private Networks](#private-networks)).

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
SwarmLLM computer takes part in regardless of this setting. Measured on a test
group whose very first request was answered by a public computer.

**To actually keep work inside a set of machines, use a device group and turn
on Private Mode:**

```toml
[pool]
private_mode_allow_lan = false   # default is true, and "LAN" includes any
                                 # other SwarmLLM computer on the same network
```

Then link the machines into a group (**More → My Devices** → create a group,
add your other devices with its invite code) and switch on **Private Mode**
there. That is an explicit list of computers, checked before any work is
handed out, and it is the only thing that decides where your prompts go. Use
both together for a private cluster: `gossip_network_id` to keep the
announcements separate, Private Mode to keep the work in.

Note `private_mode_allow_lan` is read at startup, so it belongs in the config
file before you start SwarmLLM rather than being changed while it runs.

## Firewall & internet reachability

SwarmLLM needs **TCP port 8810** (P2P primary transport) and optionally **UDP port 8800** (QUIC) open. On the same LAN, computers find each other by themselves — no ports to open. To be reachable **across the internet**, SwarmLLM tries these by itself — usually there is nothing to do:

- **UPnP** (on by default) — opens the port on a cooperative home router automatically.
- **A relay** (on by default) — if your computer can't be reached directly, even behind CGNAT, it is reached through a publicly-reachable SwarmLLM computer.
- **Manual port-forward** (TCP 8810 + UDP 8800 to your machine) plus `external_address` in config, if you want a direct connection the router won't open by itself.

If a **My Devices** invite code says *"only works on your local network,"* your computer isn't internet-reachable yet. See **[docs/NETWORKING.md](https://github.com/enapt/SwarmLLM/blob/main/docs/NETWORKING.md)** for the full guide — including the CGNAT check and how to run your own anchor node.
