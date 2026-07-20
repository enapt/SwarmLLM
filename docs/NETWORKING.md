# Networking: making your node reachable & running an anchor

SwarmLLM nodes find each other automatically **on the same local network** (via
mDNS). Connecting **across the internet** is harder because most machines sit
behind a home router that does NAT (Network Address Translation) — the outside
world can't dial into them directly. This guide explains how SwarmLLM crosses
that gap, how to check whether your node is reachable, and how to run a public
**anchor node** that helps the whole network bootstrap.

If you just want two machines on the same Wi-Fi to talk, you don't need any of
this — mDNS handles it. Read on only if you want internet-wide peering.

---

## 1. How a node becomes reachable from the internet

There are four paths, tried in roughly this order. A node needs **at least one**
to work over the internet:

| Path | What it does | Works when | Setup |
|---|---|---|---|
| **UPnP** (default on) | Asks your router to open the P2P port automatically and learns your public address | Router has UPnP/IGD enabled **and** you have a real public IP | Nothing — automatic |
| **Manual port-forward** | You forward the port on the router yourself and declare your address | You have a real public IP | Port-forward + `external_address` (below) |
| **Relay** | A publicly-reachable node forwards traffic to you | Always (even behind CGNAT), if an anchor/relay exists | Nothing on your side; someone must run a relay |
| **Hole punching (DCUtR)** | Two NAT'd nodes connect directly, coordinated by a relay | Both behind cone-NAT; needs a relay to coordinate | Nothing — automatic when a relay exists |

The invite code you generate automatically includes whichever reachable
addresses your node currently has. **If your node only has a local-network
address, the dashboard warns you** — that code will work on your LAN but not
over the internet until you set up one of the paths above.

---

## 2. Check whether you're behind CGNAT (do this first)

Carrier-Grade NAT (CGNAT) is when your **ISP** puts you behind *their* NAT, so
even port-forwarding on your own router can't make you reachable. It's the most
common reason "I forwarded the port and it still doesn't work."

1. Open your router's admin page and note its **WAN / Internet IP**.
2. On any device on your network run: `curl ifconfig.me` (or visit
   whatismyipaddress.com) to see your **actual public IP**.
3. Compare:
   - **They match**, and it's a normal public IP → 🎉 you can port-forward.
   - **They differ**, or the router's WAN IP starts with `10.`, `100.64`–
     `100.127`, or `192.168` → you're behind **CGNAT**. Port-forwarding won't
     work.

If you're behind CGNAT you have three options:
- **Call your ISP** and ask to opt out of CGNAT / get a public (static or
  dynamic) IP. Often free or a couple dollars a month.
- **Use a small VPS** (~$4–5/mo) as your anchor instead of a home box.
- **Use a relay** — you stay un-forwarded and reach the swarm through a
  publicly-reachable anchor node (see below). Slower, but no port-forwarding.

SwarmLLM will tell you if it detects CGNAT: watch the logs / dashboard for
`gateway is not routable (behind carrier-grade NAT)`.

---

## 3. Making your own node reachable

### Option A — UPnP (nothing to do)

On by default. If your router supports UPnP and you have a public IP, SwarmLLM
opens the port and confirms your public address on startup. You'll see a
"reachable across the internet" toast in the dashboard, and your invite codes
will carry the public address automatically.

Turn it off (privacy, or a router that misbehaves) with:
```toml
[network]
enable_upnp = false
```

### Option B — Manual port-forward + declared address

If UPnP is off or unavailable but you have a public IP:

1. Forward these ports on your router to the machine running SwarmLLM:
   - **TCP 8810** (P2P — this is the HTTP port `8800` **+ 10**)
   - **UDP 8800** (QUIC)
2. Tell SwarmLLM how it's reachable so it advertises the right address:
   ```toml
   [network]
   external_address = "/ip4/203.0.113.5/tcp/8810"      # your static public IP
   # or, with dynamic DNS:
   # external_address = "/dns4/myname.duckdns.org/tcp/8810"
   ```
   > Omit the trailing `/p2p/...` — the daemon appends its own peer id.

---

## 4. Running a public anchor node

An **anchor** is just an ordinary SwarmLLM node that happens to sit at a stable,
publicly-reachable address. There is **no privileged "master" role** — an anchor
has no special powers. It simply helps others by being reachable:

- new nodes **bootstrap** off it (list it in their `bootstrap_peers`),
- it answers **AutoNAT** probes so others learn their own public address,
- it acts as a **relay** so un-reachable (CGNAT) nodes can still be reached.

You only need this while the network is young. Once enough publicly-reachable
nodes exist organically, discovery self-sustains and you can retire the anchor.
A handful of stable reachable nodes is enough.

### 4.1 Where to run it

- **A VPS with a static IP** (~$4–5/mo) — most reliable, no dynamic DNS needed.
- **A home box with a real (non-CGNAT) public IP** — fine for seeding; pair a
  dynamic IP with dynamic DNS (below).

### 4.2 Dynamic DNS (for a changing home IP)

If your home IP changes, use a dynamic-DNS hostname so the address stays stable:
- **DuckDNS** — free, no domain needed, one-line updater. Simplest.
- **Cloudflare** — free if you own a domain; update an A-record via their API.

Then advertise the hostname, and libp2p re-resolves it on every dial:
```toml
[network]
external_address = "/dns4/myname.duckdns.org/tcp/8810"
```

### 4.3 Anchor setup, step by step

1. **Confirm the host is publicly reachable** (§2 — must not be CGNAT).
2. **Open the ports**: TCP 8810 + UDP 8800 (port-forward on a home router; open
   the firewall / security group on a VPS).
3. **Run the node** and set its external address:
   ```toml
   [network]
   enable_relay = true                              # (default) serve as a relay
   external_address = "/dns4/myname.duckdns.org/tcp/8810"   # or /ip4/<static-ip>/tcp/8810
   ```
4. **Get the anchor's peer id** — needed for the bootstrap multiaddr. It's
   printed at startup (`Local peer id: 12D3KooW...`) and shown on the dashboard
   identity panel, or:
   ```bash
   ./swarmllm status | grep -i "peer id"
   ```
5. **Point other nodes at it** by adding its full multiaddr to their config:
   ```toml
   [network]
   bootstrap_peers = ["/dns4/myname.duckdns.org/tcp/8810/p2p/12D3KooW...anchor-peer-id..."]
   ```
   (Or `./swarmllm run --bootstrap "/dns4/.../tcp/8810/p2p/12D3KooW..."`.)

That's it. Other nodes dial the anchor on startup, join the swarm, and — via
AutoNAT/DCUtR/relay — become discoverable to each other.

---

## 5. Troubleshooting

- **"This invite code only works on your local network"** — your node has no
  internet-reachable address yet. Enable UPnP, port-forward + set
  `external_address`, or connect through an anchor/relay.
- **UPnP does nothing** — router has UPnP disabled, or you're behind CGNAT
  (§2). Check the logs for `no IGD gateway found` (UPnP off) vs
  `gateway is not routable` (CGNAT).
- **Port-forwarded but still unreachable** — almost always CGNAT (§2). Confirm
  the router's WAN IP equals your public IP.
- **Nodes on the same Wi-Fi don't need any of this** — mDNS finds them
  automatically; if it doesn't, check that `enable_mdns = true` and that the
  machines are on the same subnet.

For a deeper look at the transport stack (Kademlia DHT, GossipSub, relay,
DCUtR) see `docs/ARCHITECTURE.md` and `docs/book/src/architecture/networking.md`.
