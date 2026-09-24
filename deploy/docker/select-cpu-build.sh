#!/bin/sh
# Installed in the CPU image as /usr/local/bin/swarmllm, so `docker run` and
# `docker exec <container> swarmllm ...` both come through here.
#
# The image carries TWO builds of the same source, and this runs the one this
# processor can execute:
#
#   x86-64-v3  AVX2 + FMA (Intel Haswell 2013, AMD Excavator 2015 and later).
#              What every x86-64 release binary is built for, and not a tuning
#              detail: candle compiles its hand-written quantized kernels only
#              under `target_feature = "avx2"`, so a build with no raised
#              target runs every quantized matmul through a scalar fallback.
#              Measured on 3B-model shapes (examples/qmatmul_bench, Q4_K and
#              Q6_K, min-of-N): 2.8-3.7x slower per generated word, 7-8.7x
#              slower reading the prompt.
#   baseline   No raised target. Runs on any x86-64 processor, slowly.
#
# Why a runtime choice rather than one build: an image runs on hosts nobody
# knows in advance. A v3-only image dies with a bare "Illegal instruction" on a
# pre-2013 processor, and on an Apple Silicon Mac whose emulator does not offer
# AVX2 (Rosetta before macOS 15). llama.cpp's and Ollama's images solve the same
# problem the same way: several CPU builds, one chosen at start.
#
# The question is asked of glibc's dynamic loader, which answers it with CPUID
# for its own glibc-hwcaps library selection (glibc >= 2.33). That is the right
# oracle for two reasons: its "x86-64-v3" is the same psABI level
# `-C target-cpu=x86-64-v3` compiles for, including the OS-support checks for
# the wider registers; and CPUID is what an emulator answers, whereas
# /proc/cpuinfo inside an emulated container describes the HOST's processor.
#
# Anything that goes wrong asking falls back to the baseline build — slow, but
# it starts.
set -eu

builds=/usr/local/lib/swarmllm
level=baseline
if /lib64/ld-linux-x86-64.so.2 --help 2>/dev/null | grep -q '^[[:space:]]*x86-64-v3 (supported'; then
    level=x86-64-v3
fi
exec "$builds/$level/swarmllm" "$@"
