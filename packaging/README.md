# SwarmLLM Packaging

This directory contains packaging files for various distribution formats.

## Homebrew (macOS / Linux)

```bash
# From the tap (when published):
brew install enapt/tap/swarmllm

# Or install from the formula file directly:
brew install --formula packaging/homebrew/swarmllm.rb
```

The formula is in `homebrew/swarmllm.rb`. To publish, create a [Homebrew tap](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap) repo at `github.com/enapt/homebrew-tap` and copy the formula there.

## Debian / Ubuntu (.deb)

```bash
# Build locally:
cargo install cargo-deb
cargo build --release
cargo deb

# Install:
sudo dpkg -i target/debian/swarmllm_*.deb
sudo systemctl enable --now swarmllm
```

Pre-built `.deb` packages are attached to [GitHub releases](https://github.com/enapt/SwarmLLM/releases).

## Fedora / RHEL (.rpm)

```bash
# Build locally:
cargo install cargo-generate-rpm
cargo build --release
cargo generate-rpm

# Install:
sudo rpm -i target/generate-rpm/swarmllm-*.rpm
sudo systemctl enable --now swarmllm
```

Pre-built `.rpm` packages are attached to [GitHub releases](https://github.com/enapt/SwarmLLM/releases).

## Arch Linux (AUR)

The `aur/PKGBUILD` builds from source. To publish to the AUR:

1. Create an AUR package at https://aur.archlinux.org/packages/swarmllm
2. Copy `PKGBUILD` and update the sha256sum
3. Generate `.SRCINFO` with `makepkg --printsrcinfo > .SRCINFO`

Users install with:
```bash
yay -S swarmllm
# or
paru -S swarmllm
```

## Systemd Service

All Linux packages install a systemd service file. After installation:

```bash
sudo systemctl enable --now swarmllm   # start on boot
sudo systemctl status swarmllm         # check status
journalctl -u swarmllm -f              # follow logs
```

Data is stored in `/var/lib/swarmllm/`. Config at `/etc/swarmllm/default.toml`.
