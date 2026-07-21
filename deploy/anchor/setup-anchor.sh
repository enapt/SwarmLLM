#!/usr/bin/env bash
#
# SwarmLLM anchor installer — run as root INSIDE a fresh Debian/Ubuntu VM.
#
#   sudo DUCKDNS_DOMAIN=your-name DUCKDNS_TOKEN=xxxx bash setup-anchor.sh
#
# Installs a hardened, non-root, bootstrap/relay-only SwarmLLM node:
#   - dedicated `swarmllm` system user (no login)
#   - latest release binary (SHA256-verified) at /usr/local/bin/swarmllm
#   - config at /etc/swarmllm/config.toml (anchor mode, relay on)
#   - DuckDNS updater (systemd timer, token stored root-only)
#   - ufw firewall: only the P2P ports open; dashboard stays loopback
#   - unattended-upgrades for OS patches
#   - systemd service (sandboxed) that auto-starts on boot
#
# Read docs/NETWORKING.md for the surrounding Proxmox/VLAN/port-forward steps.
set -euo pipefail

RAW_BASE="https://raw.githubusercontent.com/enapt/SwarmLLM/main/deploy/anchor"
REPO="enapt/SwarmLLM"
BIN="/usr/local/bin/swarmllm"
ASSET="swarmllm-linux-x86_64"          # matches src/update.rs asset naming

# --- 0. sanity ---------------------------------------------------------------
[[ $EUID -eq 0 ]] || { echo "Run as root (sudo)."; exit 1; }
[[ "$(uname -m)" == "x86_64" ]] || { echo "This installer is for x86_64."; exit 1; }
: "${DUCKDNS_DOMAIN:?Set DUCKDNS_DOMAIN (the part before .duckdns.org)}"
: "${DUCKDNS_TOKEN:?Set DUCKDNS_TOKEN (from your duckdns.org dashboard)}"
SSH_ALLOW_CIDR="${SSH_ALLOW_CIDR:-any}"   # e.g. 192.168.0.0/16 to lock SSH to LAN

echo ">> Installing packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq curl ca-certificates ufw unattended-upgrades jq >/dev/null

# --- 1. dedicated non-root user ---------------------------------------------
if ! id swarmllm &>/dev/null; then
  echo ">> Creating swarmllm system user..."
  useradd --system --no-create-home --shell /usr/sbin/nologin swarmllm
fi

# --- 2. download + verify the binary ----------------------------------------
echo ">> Fetching latest release binary ($ASSET)..."
API="https://api.github.com/repos/${REPO}/releases/latest"
DL=$(curl -fsSL "$API" | jq -r ".assets[] | select(.name==\"$ASSET\") | .browser_download_url")
SHA=$(curl -fsSL "$API" | jq -r ".assets[] | select(.name==\"${ASSET}.sha256\") | .browser_download_url")
[[ -n "$DL" && "$DL" != "null" ]] || { echo "Could not find $ASSET in the latest release."; exit 1; }
tmp=$(mktemp -d)
curl -fsSL "$DL" -o "$tmp/swarmllm"
if [[ -n "$SHA" && "$SHA" != "null" ]]; then
  curl -fsSL "$SHA" -o "$tmp/swarmllm.sha256"
  echo ">> Verifying SHA256..."
  ( cd "$tmp" && echo "$(cat swarmllm.sha256 | awk '{print $1}')  swarmllm" | sha256sum -c - )
else
  echo "!! No .sha256 sidecar found for this release — skipping checksum verify."
fi
install -m 0755 "$tmp/swarmllm" "$BIN"
rm -rf "$tmp"
"$BIN" --version || true

# --- 3. config ---------------------------------------------------------------
echo ">> Writing /etc/swarmllm/config.toml..."
install -d -m 0755 /etc/swarmllm
curl -fsSL "$RAW_BASE/config.toml" -o /etc/swarmllm/config.toml
sed -i "s#YOUR-NAME.duckdns.org#${DUCKDNS_DOMAIN}.duckdns.org#g" /etc/swarmllm/config.toml
install -d -o swarmllm -g swarmllm -m 0700 /var/lib/swarmllm

# --- 4. DuckDNS updater (token root-only, systemd timer every 5 min) ---------
echo ">> Installing DuckDNS updater..."
umask 077
cat > /etc/swarmllm/duckdns.env <<EOF
DUCKDNS_DOMAIN=${DUCKDNS_DOMAIN}
DUCKDNS_TOKEN=${DUCKDNS_TOKEN}
EOF
chmod 600 /etc/swarmllm/duckdns.env
cat > /usr/local/bin/swarmllm-duckdns.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
. /etc/swarmllm/duckdns.env
# Empty ip= lets DuckDNS use the request's source IP (our current public IP).
curl -fsS "https://www.duckdns.org/update?domains=${DUCKDNS_DOMAIN}&token=${DUCKDNS_TOKEN}&ip=" \
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

# --- 5. firewall: only the P2P ports open, dashboard stays loopback ----------
echo ">> Configuring ufw..."
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 8810/tcp comment 'SwarmLLM P2P (TCP)'
ufw allow 8800/udp comment 'SwarmLLM P2P (QUIC)'
if [[ "$SSH_ALLOW_CIDR" == "any" ]]; then
  ufw allow 22/tcp comment 'SSH'
  echo "!! SSH allowed from anywhere. Re-run with SSH_ALLOW_CIDR=192.168.0.0/16 to restrict."
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
systemctl enable --now swarmllm-duckdns.timer
systemctl start swarmllm-duckdns.service    # register the DNS record now
systemctl enable --now swarmllm-anchor.service

# --- 8. done -----------------------------------------------------------------
echo ""
echo "============================================================"
echo "  SwarmLLM anchor is up."
echo "  Hostname:  ${DUCKDNS_DOMAIN}.duckdns.org"
echo "  Status:    systemctl status swarmllm-anchor"
echo "  Logs:      journalctl -u swarmllm-anchor -f"
echo ""
echo "  Get the exact bootstrap address for your other nodes (wait ~10s first):"
echo "    KEY=\$(sudo cat /var/lib/swarmllm/api_key)"
echo "    curl -s -H \"Authorization: Bearer \$KEY\" \\"
echo "      http://127.0.0.1:8800/api/admin/network-code | jq -r '.listen_multiaddrs[]'"
echo ""
echo "  Copy the /dns4/${DUCKDNS_DOMAIN}.duckdns.org/... line into your other"
echo "  nodes' config:"
echo "    [network]"
echo "    bootstrap_peers = [\"<that line>\"]"
echo "============================================================"
