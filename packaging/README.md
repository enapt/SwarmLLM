# SwarmLLM Packaging

This directory contains packaging files for various distribution formats.

## Homebrew (macOS / Linux)

Not yet published: there is no tap, so `brew install enapt/tap/swarmllm` does not work. `homebrew/swarmllm.rb` is a template (its version is still `0.1.0` and its checksums are placeholders), so installing it directly fails too.

To publish: set the version and both `sha256` lines to a real release, create a [Homebrew tap](https://docs.brew.sh/How-to-Create-and-Maintain-a-Tap) repo at `github.com/enapt/homebrew-tap`, and copy the formula there.

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
# Build locally (an rpm version may not contain `-`, so 0.3.244-alpha is
# written 0.3.244~alpha, as release.yml does):
cargo install cargo-generate-rpm
cargo build --release
V=$(cargo metadata --no-deps --format-version 1 | python3 -c 'import json,sys; print(json.load(sys.stdin)["packages"][0]["version"])')
cargo generate-rpm --set-metadata "version = '${V//-/\~}'"

# Install:
sudo dnf install ./target/generate-rpm/swarmllm-*.rpm
sudo systemctl enable --now swarmllm
```

`rpm/` holds the package's scriptlets, the `.deb`'s maintainer scripts in rpm's
terms: `pre` creates the `swarmllm` user, `post` the data folder and its
config, `posttrans` restarts a node that was running when it was upgraded,
`preun` stops and disables it on removal only. The user and
`/var/lib/swarmllm` are kept when the package is removed. The binary needs
glibc 2.35, so RHEL 9 and its rebuilds refuse the package (FUTURE_WORK #311).

Pre-built `.rpm` packages are attached to [GitHub releases](https://github.com/enapt/SwarmLLM/releases).

## Arch Linux (AUR)

Not yet published: there is no `swarmllm` package on the AUR, so `yay -S swarmllm` does not work. `aur/PKGBUILD` builds from source (its `pkgver` is still `0.1.0`). To publish it:

1. Create an AUR package at https://aur.archlinux.org/packages/swarmllm
2. Copy `PKGBUILD`, set `pkgver` to a real release, and update the sha256sum
3. Generate `.SRCINFO` with `makepkg --printsrcinfo > .SRCINFO`

Once published, users install with `yay -S swarmllm` or `paru -S swarmllm`.

## Systemd Service

All Linux packages install a systemd service file. After installation:

```bash
sudo systemctl enable --now swarmllm   # start on boot
sudo systemctl status swarmllm         # check status
journalctl -u swarmllm -f              # follow logs
```

Data is stored in `/var/lib/swarmllm/`, and the node reads its settings from
`/var/lib/swarmllm/config.toml` — the file the dashboard saves to. Both
packages copy `/etc/swarmllm/default.toml` there on first install only, so
editing the `/etc` copy afterwards changes nothing.
