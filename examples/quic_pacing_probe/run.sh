#!/bin/bash
# QUIC pacing probe: Cubic vs BBR for a split's bursts. Needs no root:
#   cd examples/quic_pacing_probe && cargo build --release --offline
#   unshare -rn ./run.sh "rate 20mbit loss 1%"     # netem arguments after the delay
# 2026-10-20, 100 ms round trip, 20 Mbit/s (floor ~119 ms for 43 KB, ~214 for 280 KB):
#   Cubic 154 / 219 ms, with 1% loss 149 / 493; BBR 119 / 219, with 1% loss 119 / 219.
B="$(dirname "$0")/target/release/quicprobe"
ip link set lo up
tc qdisc add dev lo root netem delay 50ms ${1:-}
echo "netem: delay 50ms ${1:-}; $(ping -c 2 -q 127.0.0.1 | tail -1)"
for cc in cubic bbr; do
  CC=$cc "$B" 43000 300 22 0 0 | sed "s/^/$cc /"     # one 5-token check of a 70B, Q8_0
  CC=$cc "$B" 280000 400 22 0 0 | sed "s/^/$cc /"    # a 32-row guess tree of a 70B
done
