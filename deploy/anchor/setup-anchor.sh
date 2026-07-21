#!/usr/bin/env bash
#
# SwarmLLM anchor installer — run as root on a fresh Debian/Ubuntu host
# (a VPS with a static IP, or a home VM on a non-CGNAT connection).
#
# Pick how the node advertises itself, then run:
#
#   # VPS with a static public IP + a human-readable DuckDNS name (recommended):
#   sudo PUBLIC_IP=203.0.113.5 DUCKDNS_DOMAIN=my-swarm DUCKDNS_TOKEN=xxxx bash setup-anchor.sh
#
#   # VPS static IP only (no DNS account):
#   sudo PUBLIC_IP=203.0.113.5 bash setup-anchor.sh
#
#   # Home box, dynamic (non-CGNAT) IP — DuckDNS auto-tracks it:
#   sudo DUCKDNS_DOMAIN=my-swarm DUCKDNS_TOKEN=xxxx bash setup-anchor.sh
#
# Optional:
#   SSH_ALLOW_CIDR=1.2.3.4/32   restrict SSH to an IP/subnet (default: any)
#   SKIP_DOWNLOAD=1             keep an already-placed /usr/local/bin/swarmllm.
#                              Use when you built the anchor-capable binary
#                              yourself and scp'd it (the current published
#                              release may predate --anchor).
#
# Installs a hardened, non-root, bootstrap/relay-only node. See
# deploy/anchor/README.md + docs/NETWORKING.md.
set -euo pipefail

RAW_BASE="https://raw.githubusercontent.com/enapt/SwarmLLM/main/deploy/anchor"
REPO="enapt/SwarmLLM"
BIN="/usr/local/bin/swarmllm"
ASSET="swarmllm-linux-x86_64"          # matches src/update.rs asset naming
P2P_TCP=8810                            # listen_port (8800) + 10
P2P_UDP=8800

# --- 0. inputs / sanity ------------------------------------------------------
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
[[ "$(uname -m)" == "x86_64" ]] || { echo "This installer is for x86_64."; exit 1; }
PUBLIC_IP="${PUBLIC_IP:-}"
DUCKDNS_DOMAIN="${DUCKDNS_DOMAIN:-}"
DUCKDNS_TOKEN="${DUCKDNS_TOKEN:-}"
SSH_ALLOW_CIDR="${SSH_ALLOW_CIDR:-any}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# Decide the advertised external address. A DuckDNS name (human-readable +
# portable) wins when present; otherwise advertise the raw static IP.
if [[ -n "$DUCKDNS_DOMAIN" ]]; then
  [[ -n "$DUCKDNS_TOKEN" ]] || { echo "DUCKDNS_DOMAIN set but DUCKDNS_TOKEN missing."; exit 1; }
  HOST="${DUCKDNS_DOMAIN}.duckdns.org"
  PROTO="dns4"
elif [[ -n "$PUBLIC_IP" ]]; then
  HOST="$PUBLIC_IP"
  PROTO="ip4"
else
  echo "Set PUBLIC_IP (static IP) and/or DUCKDNS_DOMAIN+DUCKDNS_TOKEN (DNS name)."; exit 1
fi
# Advertise the host on both transports (TCP + QUIC). Single-line TOML array so
# the sed rewrite below is a one-liner.
EXTERNAL_ADDRS="[\"/${PROTO}/${HOST}/tcp/${P2P_TCP}\", \"/${PROTO}/${HOST}/udp/${P2P_UDP}/quic-v1\"]"
echo ">> Advertising external addresses: ${EXTERNAL_ADDRS}"

echo ">> Installing packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates ufw unattended-upgrades jq >/dev/null

# --- 1. dedicated non-root user ---------------------------------------------
if ! id swarmllm &>/dev/null; then
  echo ">> Creating swarmllm system user..."
  useradd --system --no-create-home --shell /usr/sbin/nologin swarmllm
fi

# --- 2. binary ---------------------------------------------------------------
if [[ "$SKIP_DOWNLOAD" == "1" ]]; then
  [[ -x "$BIN" ]] || { echo "SKIP_DOWNLOAD=1 but $BIN is missing. scp your binary there first."; exit 1; }
  echo ">> Using existing binary at $BIN (SKIP_DOWNLOAD=1)."
else
  echo ">> Fetching latest release binary ($ASSET)..."
  META=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest")
  DL=$(echo "$META" | jq -r ".assets[] | select(.name==\"$ASSET\") | .browser_download_url")
  SHA=$(echo "$META" | jq -r ".assets[] | select(.name==\"${ASSET}.sha256\") | .browser_download_url")
  [[ -n "$DL" && "$DL" != "null" ]] || {
    echo "No $ASSET in the latest release. Build the anchor binary on your dev box"
    echo "(RUSTFLAGS=\"\" cargo build --release), scp it to $BIN, and re-run with SKIP_DOWNLOAD=1."
    exit 1
  }
  tmp=$(mktemp -d)
  curl -fsSL "$DL" -o "$tmp/swarmllm"
  if [[ -n "$SHA" && "$SHA" != "null" ]]; then
    curl -fsSL "$SHA" -o "$tmp/swarmllm.sha256"
    echo ">> Verifying SHA256..."
    ( cd "$tmp" && echo "$(awk '{print $1}' swarmllm.sha256)  swarmllm" | sha256sum -c - )
  else
    echo "!! No .sha256 sidecar — skipping checksum verify."
  fi
  install -m 0755 "$tmp/swarmllm" "$BIN"
  rm -rf "$tmp"
fi

# Fail fast if this binary predates anchor mode (else the service would crash-loop
# on an "unexpected argument --anchor").
if ! "$BIN" run --help 2>/dev/null | grep -q -- '--anchor'; then
  echo "!! This swarmllm binary does not support --anchor (predates R143)."
  echo "   On your dev box:  RUSTFLAGS=\"\" cargo build --release"
  echo "   Then:             scp target/release/swarmllm root@<host>:$BIN"
  echo "   And re-run this installer with SKIP_DOWNLOAD=1."
  exit 1
fi
"$BIN" --version || true

# --- 3. config ---------------------------------------------------------------
echo ">> Writing /etc/swarmllm/config.toml..."
install -d -m 0755 /etc/swarmllm
curl -fsSL "$RAW_BASE/config.toml" -o /etc/swarmllm/config.toml
sed -i "s#^external_addresses = .*#external_addresses = ${EXTERNAL_ADDRS}#" /etc/swarmllm/config.toml
install -d -o swarmllm -g swarmllm -m 0700 /var/lib/swarmllm

# --- 4. DuckDNS updater (only when a DuckDNS name was given) -----------------
if [[ -n "$DUCKDNS_DOMAIN" ]]; then
  echo ">> Installing DuckDNS updater..."
  umask 077
  cat > /etc/swarmllm/duckdns.env <<EOF
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
DUCKDNS_IP=${PUBLIC_IP}
EOF
  chmod 600 /etc/swarmllm/duckdns.env
  cat > /usr/local/bin/swarmllm-duckdns.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
. /etc/swarmllm/duckdns.env
# DUCKDNS_IP empty -> DuckDNS uses the request source IP (dynamic tracking).
# Set (static VPS IP) -> pins the record to it.
curl -fsS "https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=${DUCKDNS_IP}" \
  | grep -q '^OK$' || { echo "DuckDNS update failed"; exit 1; }
EOF
  chmod 700 /usr/local/bin/swarmllm-duckdns.sh
  cat > /etc/systemd/system/swarmllm-duckdns.service <<'EOF'
[Unit]
Description=Update DuckDNS record for the SwarmLLM anchor
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/swarmllm-duckdns.sh
EOF
  cat > /etc/systemd/system/swarmllm-duckdns.timer <<'EOF'
[Unit]
Description=Refresh DuckDNS every 5 minutes
[Timer]
OnBootSec=30s
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
EOF
fi

# --- 5. firewall: only the P2P ports open, dashboard stays loopback ----------
echo ">> Configuring ufw..."
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow ${P2P_TCP}/tcp comment 'SwarmLLM P2P (TCP)'
ufw allow ${P2P_UDP}/udp comment 'SwarmLLM P2P (QUIC)'
if [[ "$SSH_ALLOW_CIDR" == "any" ]]; then
  ufw allow 22/tcp comment 'SSH'
  echo "!! SSH allowed from anywhere. Re-run with SSH_ALLOW_CIDR=<your-ip>/32 to restrict."
else
  ufw allow from "$SSH_ALLOW_CIDR" to any port 22 proto tcp comment 'SSH (restricted)'
fi
ufw --force enable

# --- 6. OS auto-patching -----------------------------------------------------
echo ">> Enabling unattended-upgrades..."
dpkg-reconfigure -f noninteractive unattended-upgrades >/dev/null 2>&1 || true
systemctl enable --now unattended-upgrades >/dev/null 2>&1 || true

# --- 7. systemd service ------------------------------------------------------
echo ">> Installing systemd service..."
curl -fsSL "$RAW_BASE/swarmllm-anchor.service" -o /etc/systemd/system/swarmllm-anchor.service
systemctl daemon-reload
if [[ -n "$DUCKDNS_DOMAIN" ]]; then
  systemctl enable --now swarmllm-duckdns.timer
  systemctl start swarmllm-duckdns.service    # register the DNS record now
fi
systemctl enable --now swarmllm-anchor.service

# --- 8. done -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SwarmLLM anchor is up."
echo "  Advertising: ${EXTERNAL_ADDRS}"
echo "  Status:      systemctl status swarmllm-anchor"
echo "  Logs:        journalctl -u swarmllm-anchor -f"
echo ""
echo "  Get the exact bootstrap address for your other nodes (wait ~10s):"
echo "    KEY=\$(sudo cat /var/lib/swarmllm/api_key)"
echo "    curl -s -H \"Authorization: Bearer \$KEY\" \\"
echo "      http://127.0.0.1:8800/api/admin/network-code | jq -r '.listen_multiaddrs[]'"
echo ""
echo "  Copy the ${HOST} line into your other nodes' config:"
echo "    [network]"
echo "    bootstrap_peers = [\"<that line>\"]"
echo ""
echo "  NOTE (VPS): also open TCP ${P2P_TCP} + UDP ${P2P_UDP} in your provider's"
echo "  firewall panel (e.g. IONOS Cloud Panel) — it sits in front of ufw."
echo "============================================================"
