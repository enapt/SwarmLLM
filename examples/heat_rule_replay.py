#!/usr/bin/env python3
"""Replay the heat guard's PROCESSOR rule over a recorded temperature log.

The guard (`src/daemon/host_heat.rs`) holds back work for other computers once
the processor has read at or above PROCESSOR_HOT_C for PROCESSOR_HOT_FOR with no
reading under the line in between (a reading under it starts the clock again),
and lets it go once it has read at or under PROCESSOR_COOL_C for COOL_FOR. This
asks how often that rule WOULD have held on a real machine, and under what load
— the calibration #440 stage D asked for before a WSL2 node acted on Windows'
thermal zone (FUTURE_WORK #440 (D), 2026-11-18: 313 h of this laptop's zone,
14 holds, 2.4 h, each under load).

Input: a CSV with columns `utc` (ISO time), `thermal_c` and `cpu_pct` — what the
safety kit's Windows telemetry writes (`/mnt/c/temp/gate-telemetry/*-win.csv`).
Readings are thinned to one per `--every` seconds, the guard's round (30 s);
a gap over 10 minutes (a reboot, the logger off) forgets any clock or hold.

usage: python3 -I examples/heat_rule_replay.py <telemetry.csv> [--every 30]
       [--hot 93] [--cool 88] [--hot-for 300] [--cool-for 60]

Keep the defaults in step with the constants in host_heat.rs.
"""

import argparse
import csv
import datetime as dt
import sys

GAP_FORGETS_S = 600


def readings(path, every):
    out, last = [], None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = dt.datetime.fromisoformat(row["utc"])
                zone = float(row["thermal_c"])
                cpu = float(row["cpu_pct"])
            except (KeyError, TypeError, ValueError):
                continue
            if last is None or (t - last).total_seconds() >= every:
                out.append((t, zone, cpu))
                last = t
    return out


def holds(rows, hot, cool, hot_for, cool_for):
    """Each hold as (start, end, how it ended)."""
    found, clock, held, cool_since, start, prev = [], None, False, None, None, None
    for t, zone, _ in rows:
        if prev is not None and (t - prev).total_seconds() > GAP_FORGETS_S:
            if held:
                found.append((start, prev, "gap"))
            clock, held, cool_since = None, False, None
        prev = t
        if not held:
            if zone >= hot:
                clock = clock or t
                if (t - clock).total_seconds() >= hot_for:
                    held, start, cool_since = True, t, None
            else:
                clock = None
        elif zone <= cool:
            cool_since = cool_since or t
            if (t - cool_since).total_seconds() >= cool_for:
                found.append((start, t, "cooled"))
                held, clock, cool_since = False, None, None
        else:
            cool_since = None
    if held:
        found.append((start, prev, "open"))
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--every", type=int, default=30)
    ap.add_argument("--hot", type=float, default=93.0)
    ap.add_argument("--cool", type=float, default=88.0)
    ap.add_argument("--hot-for", type=int, default=300)
    ap.add_argument("--cool-for", type=int, default=60)
    a = ap.parse_args()
    rows = readings(a.csv, a.every)
    if not rows:
        sys.exit("no readings")
    span_h = (rows[-1][0] - rows[0][0]).total_seconds() / 3600
    busy_h = sum(1 for _, _, c in rows if c >= 40) * a.every / 3600
    found = holds(rows, a.hot, a.cool, a.hot_for, a.cool_for)
    held_h = sum((e - s).total_seconds() for s, e, _ in found) / 3600
    print(f"{len(rows)} readings, one per {a.every} s, over {span_h:.0f} h "
          f"({rows[0][0]} -> {rows[-1][0]}); processor 40%+ busy {busy_h:.1f} h")
    print(f"holds: {len(found)}, {held_h:.2f} h in all")
    for s, e, why in found:
        before = [c for t, _, c in rows if s - dt.timedelta(seconds=a.hot_for) <= t <= s]
        print(f"  {s}  {(e - s).total_seconds() / 60:6.1f} min  {why:6s}  "
              f"busy before {min(before):.0f}-{max(before):.0f}% "
              f"(mean {sum(before) / len(before):.0f}%)")


if __name__ == "__main__":
    main()
