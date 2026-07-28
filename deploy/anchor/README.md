# Running a SwarmLLM anchor node

An **anchor** is an ordinary SwarmLLM node that sits at a stable, publicly-reachable
address and helps the network bootstrap: new nodes dial it to join, it answers
AutoNAT probes so peers learn their own public address, and it relays for peers
behind NAT. There is **no privileged "master" role** — an anchor has no special
powers; retiring it changes nothing about how the network works.

This directory sets up a **hardened, inference-free** anchor: it runs the daemon
with `--anchor`, so no models load, no HuggingFace/shard downloads happen, no
inference subprocess ever spawns, and the dashboard binds to loopback only. The
only thing reachable from the internet is the P2P transport.

## Prerequisites

- **A SwarmLLM release that includes anchor mode** (`--anchor`, added in R143).
  The installer downloads the latest GitHub release binary — if that release
  predates anchor mode, the service will fail with "unexpected argument
  --anchor". Cut a release from `main` first, or build the binary elsewhere and
  place it at `/usr/local/bin/swarmllm` before running the installer. (Building
  from source *on* a 512 MB anchor VM isn't practical — candle needs more RAM to
  compile; build on your dev box and copy the binary over.)
- A host with a **real public IP** (not CGNAT — check: router WAN IP must equal
  `curl -4 ifconfig.me`). **A cheap VPS (static IP) is the simplest + safest** —
  no port-forwarding, no dynamic DNS, and your home network is never exposed.
  A home box on a non-CGNAT connection also works.
- **How the node advertises itself** — pick one (the installer takes both):
  - **Static IP** (VPS): pass `PUBLIC_IP=<your-static-ip>`. No DNS account needed.
  - **DuckDNS name** (readable + portable): a free hostname + token from
    duckdns.org. Recommended even with a static IP — if you ever move hosts you
    update one DNS record instead of every peer's config.
- For a home host with a **dynamic** IP: use DuckDNS alone (it auto-tracks the IP).

## 1. Create an isolated VM (Proxmox)

Run the anchor in its **own VM** — never on the Proxmox host or beside other
services. A full VM (not an LXC) is preferred for internet-exposed software: it
has a stronger isolation boundary. A minimal Debian 12 / Ubuntu Server (no GUI,
1 vCPU, 512 MB–1 GB RAM, 4 GB disk) is plenty.

## 2. Network isolation (the important part)

Put the VM on an **isolated bridge/VLAN** and add firewall rules (Proxmox
Datacenter → Firewall, or your router/switch) so a compromised anchor can't
pivot into your network:

- **Allow inbound**: TCP 8810, UDP 8800 (from the port-forward).
- **Allow outbound**: internet (peers, GitHub updates, DuckDNS).
- **Block** the VM from reaching your LAN subnet and the Proxmox management
  interface (`:8006`, SSH to other hosts). This containment is what makes home
  hosting safe.

## 3. Port-forward (home hosts)

Forward to the VM's LAN IP:

- **TCP 8810** → anchor (P2P primary transport)
- **UDP 8800** → anchor (QUIC)

**Do not** forward TCP 8800 — that's the dashboard, and it stays loopback-only.

## 4. Run the installer

Copy `setup-anchor.sh` onto the host and run it as root. Pick the invocation
that matches your setup:

```bash
# VPS: static IP + a readable DuckDNS name pinned to it (recommended)
sudo PUBLIC_IP=203.0.113.5 DUCKDNS_DOMAIN=your-name DUCKDNS_TOKEN=xxxx bash setup-anchor.sh

# VPS: static IP only (no DNS account)
sudo PUBLIC_IP=203.0.113.5 bash setup-anchor.sh

# Home box, dynamic non-CGNAT IP (DuckDNS auto-tracks it)
sudo DUCKDNS_DOMAIN=your-name DUCKDNS_TOKEN=xxxx bash setup-anchor.sh
```

Optional env vars:
- `SSH_ALLOW_CIDR=1.2.3.4/32` — restrict SSH to your IP/subnet (recommended).
- `SKIP_DOWNLOAD=1` — keep a binary you already placed at
  `/usr/local/bin/swarmllm` (see the version note above — until a release ships
  anchor mode, build it on your dev box and `scp` it over, then use this).

**VPS firewall:** most providers (IONOS Cloud Panel, etc.) have their own
firewall *in front of* the VM. Open **TCP 8810 + UDP 8800** there too — the
installer's `ufw` only covers the OS-level firewall.

The script: creates a non-root `swarmllm` user; downloads + SHA256-verifies the
latest release binary; writes `/etc/swarmllm/config.toml` (anchor mode, relay on,
your DuckDNS host in `external_addresses`); installs a DuckDNS updater
(systemd timer, token stored root-only); sets a default-deny `ufw` firewall
(only the P2P ports open); enables `unattended-upgrades`; and installs a
**sandboxed** systemd service (see `swarmllm-anchor.service`) that starts on boot.

## 5. Connect your other nodes

Once the anchor has been up ~10s, ask it for its exact dial address (the API is
loopback-only, so run this **on the anchor box**):

```bash
KEY=$(sudo cat /var/lib/swarmllm/api_key)
curl -s -H "Authorization: Bearer $KEY" \
  http://127.0.0.1:8800/api/admin/network-code | jq -r '.listen_multiaddrs[]'
```

That prints your ready-to-copy bootstrap string(s), e.g.
`/dns4/your-name.duckdns.org/tcp/8810/p2p/12D3KooW...`. Put it on every *other*
node:

```toml
[network]
bootstrap_peers = ["/dns4/your-name.duckdns.org/tcp/8810/p2p/12D3KooW..."]
```

They dial the anchor on startup, join the swarm, and — via AutoNAT/DCUtR/relay —
become discoverable to each other.

## What's hardened

- Runs as a **non-root** system user, no login shell.
- systemd sandbox: `ProtectSystem=strict`, `NoNewPrivileges`, `PrivateTmp`,
  kernel/clock/hostname protections, `SystemCallFilter=@system-service`,
  read-only filesystem except `/var/lib/swarmllm`.
- Dashboard/API **loopback-only** (reach it via `ssh -L 8800:localhost:8800`).
- Default-deny firewall; only TCP 8810 + UDP 8800 exposed.
- Inference surface **not running at all** (`--anchor`).
- OS auto-patched (`unattended-upgrades`).
- **swarmllm auto-updated by a root-run timer** (`swarmllm-update.timer`, daily),
  *not* by the daemon. The sandboxed non-root daemon deliberately can't rewrite
  its own binary, so a separate root-privileged updater checks for a newer
  release, verifies the SHA256, swaps the binary and restarts the service — the
  same split as `unattended-upgrades` for the OS. Refuses unverified binaries;
  never downgrades.

## Updating an anchor that is already running

The automatic updater replaces the **binary** and restarts the service. It does
not touch its own systemd units — not `swarmllm-anchor.service`, not
`swarmllm-update.timer`, not `swarmllm-update.sh` itself.

That is deliberate (a bad unit file could break both the anchor and the thing
meant to repair it), but it has a consequence worth knowing: **changes to how
often the anchor checks for updates, or to how it runs, never reach an anchor
that is already deployed.** The binary keeps updating; everything around it
stays as installed.

Concretely, the check interval moved from daily to hourly in v0.3.44. An anchor
installed before that keeps checking daily however many times its binary
updates. During alpha, when several releases can ship in a day, that means it is
usually running something other than the current build.

To pick up unit changes on an existing anchor, refresh them once:

```bash
RAW=https://raw.githubusercontent.com/enapt/SwarmLLM/main/deploy/anchor
curl -fsSL "$RAW/swarmllm-update.sh"    -o /usr/local/bin/swarmllm-update.sh
chmod +x /usr/local/bin/swarmllm-update.sh
curl -fsSL "$RAW/swarmllm-update.timer" -o /etc/systemd/system/swarmllm-update.timer
curl -fsSL "$RAW/swarmllm-anchor.service" -o /etc/systemd/system/swarmllm-anchor.service
systemctl daemon-reload
systemctl restart swarmllm-update.timer swarmllm-anchor.service
systemctl list-timers swarmllm-update.timer   # confirm the new cadence
```

Check what an anchor is actually running with `swarmllm --version`, and when it
last looked with `journalctl -u swarmllm-update --since -1d`.

## Maintenance

- **Status / logs**: `systemctl status swarmllm-anchor`, `journalctl -u swarmllm-anchor -f`
- **Update binary now** (instead of waiting for the daily timer):
  `sudo systemctl start swarmllm-update.service` then `journalctl -u swarmllm-update`.
- **Check the auto-updater**: `systemctl list-timers swarmllm-update.timer`.
- **Back up** `/var/lib/swarmllm` (holds the node identity keypair + credit
  balance) and snapshot the VM so you can roll back cleanly.
- **Retiring it**: once enough publicly-reachable nodes exist organically, the
  network self-sustains and you can stop the anchor — nothing else depends on it.
