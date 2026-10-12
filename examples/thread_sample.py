#!/usr/bin/env python3
"""What is every thread of a node (and its model processes) doing, every 50 ms?

For a node that stalls without using the CPU: is a Tokio runtime worker burning CPU (synchronous
work on the runtime) or blocked in a syscall (a sleep, file I/O, a lock), and which? Written for
FUTURE_WORK #244, where it showed a runtime worker in clock_nanosleep with an nvidia-smi child alive
— not the task the stall's timing pointed at. docs/DIAGNOSTICS.md § "Which thread is a stalled node
sitting in" is the method.

    thread_sample.py PID SECONDS > out.txt
    thread_sample.py --spawn 'MATCH' SECONDS CMD ARGS... > out.txt   # run CMD, sample the first swarmllm
                                                                      # whose command line contains MATCH,
                                                                      # then wait for CMD

Each line: UTC time, then every thread that used CPU in the interval or is not idle, as
name/tid:+ticks:state:syscall[:wchan]. `syscall` is the number from /proc/.../syscall ("run" while on
CPU; x86-64: 230 clock_nanosleep, 202 futex, 232 epoll_wait); idle Tokio workers park in futex or
epoll_wait. Tokio's runtime workers and its blocking-pool threads are BOTH named tokio-rt-worker.
/proc/.../syscall needs ptrace-attach permission, which yama ptrace_scope 1 grants only to an
ANCESTOR — so start the node (or the rig that starts it) from here with --spawn, or it reads "?"
(wchan is readable either way)."""
import os, sys, time, datetime, subprocess

child = None
if sys.argv[1] == "--spawn":
    match, secs = sys.argv[2], float(sys.argv[3])
    child = subprocess.Popen(sys.argv[4:])
    pid = None
    for _ in range(3000):
        for p in os.listdir("/proc"):
            if p.isdigit():
                try:
                    comm = open(f"/proc/{p}/comm").read().strip()
                    cmd = open(f"/proc/{p}/cmdline").read().replace("\0", " ")
                except OSError:
                    continue
                if comm == "swarmllm" and match in cmd:
                    pid = int(p)
                    break
        if pid:
            break
        time.sleep(0.1)
    if not pid:
        print("no process matched", match, flush=True)
        sys.exit(child.wait())
    print("sampling pid", pid, flush=True)
else:
    pid, secs = int(sys.argv[1]), float(sys.argv[2])
IDLE = {"202", "232", "281", "7", "271"}  # futex, epoll_wait, epoll_pwait, poll, ppoll


def pids(root):
    out = [root]
    try:
        for t in os.listdir(f"/proc/{root}/task"):
            for c in open(f"/proc/{root}/task/{t}/children").read().split():
                out += pids(int(c))
    except OSError:
        pass
    return sorted(set(out))


def read(path):
    try:
        return open(path).read()
    except OSError:
        return ""


def snap():
    s = {}
    for p in pids(pid):
        try:
            tids = os.listdir(f"/proc/{p}/task")
        except OSError:
            continue
        for t in tids:
            base = f"/proc/{p}/task/{t}"
            raw = read(base + "/stat")
            if not raw:
                continue
            name = raw[raw.index("(") + 1:raw.rindex(")")]
            f = raw[raw.rindex(")") + 2:].split()
            sc = read(base + "/syscall").split()
            sc = "run" if sc and sc[0] == "running" else (sc[0] if sc else "?")
            wchan = read(base + "/wchan").strip()
            s[(p, int(t))] = (name, f[0], int(f[11]) + int(f[12]), sc, wchan)
    return s


end = time.time() + secs
prev = snap()
while time.time() < end:
    time.sleep(0.05)
    cur = snap()
    out = []
    for k, (name, st, ticks, sc, wchan) in sorted(cur.items()):
        d = ticks - prev.get(k, (name, st, ticks, sc, wchan))[2]
        if d > 0 or st in ("R", "D") or sc not in IDLE:
            w = f":{wchan}" if wchan and wchan != "0" else ""
            out.append(f"{name}/{k[1]}:+{d}:{st}:{sc}{w}")
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(ts, " ".join(out), flush=True)
    prev = cur
if child:
    sys.exit(child.wait())
