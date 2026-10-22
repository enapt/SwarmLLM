// SwarmLLM: llama.cpp's quantized matmul (MMQ) launched without a ggml context.
//
// Everything under `include/` and `src/` is llama.cpp's ggml, copied verbatim
// from commit 4b1a27fa0eb8 (2026-09-25; MIT, see LICENSE). This file replaces
// only the HOST side of `launch_mul_mat_q` / `mul_mat_q_switch_J` /
// `mul_mat_q_case` (ggml-cuda/mmq.cuh), which reach for a
// `ggml_backend_cuda_context` (its memory pool, its device table): here the
// caller hands in the stream and the stream-k fixup buffer, and a failure is
// a return code, never `abort()` in a worker serving requests.
//
// Why it exists: candle's MMQ is llama.cpp's OLD dp4a kernel with a fixed
// 64-column tile. On an RTX 3070, a Qwen2.5 7B Q4_K_M forward of 16 rows cost
// 83.3 ms with it and 22.9 ms in llama.cpp; 32 rows 84.4 vs 25.8; 64 rows
// 107.8 (cuBLAS) vs 37.4 (2026-10-21, ~/swarmllm-318/rowcost.sh). That width
// is a split's tree check, on every machine of the chain, and every prompt
// chunk. llama.cpp runs this kernel for every batch above 8 rows on a card
// with int8 tensor cores (`ggml_cuda_should_use_mmq`: `turing_mma_available`).
//
// Everything here sits in `namespace swarm_ggml`. A `cuda` build also links
// llama-cpp-sys-2's own ggml-cuda (the `llama` feature), a different commit:
// its quantizers and `ggml_abort` collide with these as strong symbols (a
// link error), and its kernel templates as WEAK ones — merged without a word,
// so one copy's launch could run the other's device code. The namespace gives
// every C++ symbol here a name of its own; `ggml_abort`, the one C-linkage
// function the copied code calls, is renamed below.
#pragma once

// The system and CUDA headers the copied ones include, OUTSIDE the namespace:
// each has an include guard, so the includes inside it add nothing.
#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// What `ggml-cuda/vendors/cuda.h` includes, under the same conditions.
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#endif
#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif

#define ggml_abort swarm_ggml_abort

namespace swarm_ggml {

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/mmq.cuh"

// What a launch needs to know about the current device. llama.cpp keeps it in
// `ggml_cuda_info()`; here it is read once per device.
struct swarm_mmq_device {
    int    cc;        // 100 * major + 10 * minor, llama.cpp's encoding
    int    nsm;       // streaming multiprocessors
    size_t smpbo;     // shared memory per block, opt-in
    int    warp_size;
};

// The failures a launch reports instead of aborting.
enum swarm_mmq_status : int {
    SWARM_MMQ_OK              = 0,
    SWARM_MMQ_UNSUPPORTED     = 1, // type or shape this kernel does not cover
    SWARM_MMQ_FIXUP_TOO_SMALL = 2, // the caller's fixup buffer is under `need`
    SWARM_MMQ_CUDA_ERROR      = 3, // a runtime call failed
    SWARM_MMQ_NO_TILE         = 4, // no tile width fits this card's shared memory
};

// `cudaGetDevice` + attributes, cached per device the way `ggml_cuda_info` is.
swarm_mmq_status swarm_mmq_current_device(swarm_mmq_device & out);

// A dense product dst[ncols_y][nrows_x] = y · xᵀ: x is `nrows_x` rows of
// `ncols_x` quantized values (GGUF layout, `stride_row_x` blocks a row), y is
// the activation already in llama.cpp's `block_q8_1_mmq` layout
// (`quantize_mmq_q8_1_cuda`), dst is row-major with `nrows_x` floats a row.
struct swarm_mmq_dense {
    const char * x;
    const int  * y;
    float      * dst;
    int64_t      ncols_x;
    int64_t      nrows_x;
    int64_t      ncols_y;
    int64_t      stride_row_x;
};

// llama.cpp's `mmq_args` for a plain 2-D product: one channel, one sample.
static mmq_args swarm_mmq_args(const swarm_mmq_dense & p, ggml_type type) {
    // Stride of y between channels, in ints — unused with one channel, but
    // set as `ggml_cuda_mul_mat_q` sets it.
    const int64_t ncols_x_padded = GGML_PAD(p.ncols_x, MATRIX_ROW_PADDING);
    const int64_t s12 = p.ncols_y * ncols_x_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const mmq_args args = {
        p.x, type, p.y, nullptr, nullptr, p.dst,
        nullptr,
        p.ncols_x, p.nrows_x, p.ncols_y, p.stride_row_x, p.ncols_y, p.nrows_x,
        1, 1, p.nrows_x * p.stride_row_x, s12, p.nrows_x * p.ncols_y,
        1, 1, p.nrows_x * p.stride_row_x, s12, p.nrows_x * p.ncols_y,
        p.ncols_y, p.ncols_y};
    return args;
}

// `launch_mul_mat_q` with the pool and the device table taken out. With
// `fixup == nullptr` and `launch == false` it only reports, through `need`,
// how many floats of fixup buffer the launch will want (0 when none) — and
// raises the kernel's shared-memory limit, the first time, so that a product
// planned once outside a CUDA graph capture enqueues nothing but kernels
// inside one (`swarm_mmq_prepare_few_rows`).
template <ggml_type type, int J, bool fallback>
static swarm_mmq_status swarm_mmq_launch(
        const swarm_mmq_device & dev, const mmq_args & args, float * fixup, size_t fixup_floats,
        size_t & need, bool launch, cudaStream_t stream) {
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, dev.cc);
    if (config.nthreads % dev.warp_size != 0) {
        return SWARM_MMQ_UNSUPPORTED;
    }
    const int nwarps        = config.nthreads / dev.warp_size;
    const int nbytes_shared = mmq_get_nbytes_shared(config, dev.cc);
    const dim3 block_dims(dev.warp_size, nwarps, 1);

    const int nty  = (args.nrows_x   + config.I - 1) / config.I;
    const int ntx  = (args.ncols_max + config.J - 1) / config.J;
    const int ntzw = args.nchannels_y * args.nsamples_y;

    const uint3 blocks_per_ne00_fd = init_fastdiv_values(args.ncols_x / ggml_cuda_type_traits<type>::qk);

    const int  ntiles_dst      = ntx * nty * ntzw;
    const int  tiles_nwaves    = (ntiles_dst + dev.nsm - 1) / dev.nsm;
    const int  tiles_efficiency_percent = 100 * ntiles_dst / (dev.nsm * tiles_nwaves);
    const int  nblocks_stream_k = GGML_CUDA_CC_IS_NVIDIA(dev.cc) && tiles_efficiency_percent >= 90 ? ntiles_dst : dev.nsm;
    const bool fixup_needed     = config.stream_k && ntiles_dst % nblocks_stream_k != 0;
    need = fixup_needed ? size_t(nblocks_stream_k) * config.J * config.I : 0;

    // Raised once per kernel and device, as CUDA_SET_SHARED_MEMORY_LIMIT does,
    // but a failure is reported rather than aborting the process. Before the
    // planning call returns, not only before a launch: it is the one driver
    // call here besides the launches, and a capture must not see it.
    {
        static bool raised[GGML_CUDA_MAX_DEVICES] = { false };
        int id = 0;
        if (cudaGetDevice(&id) != cudaSuccess || id < 0 || id >= GGML_CUDA_MAX_DEVICES) {
            return SWARM_MMQ_CUDA_ERROR;
        }
        if (!raised[id]) {
            if (cudaFuncSetAttribute(mul_mat_q<type, J, fallback>, cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared) != cudaSuccess) {
                return SWARM_MMQ_CUDA_ERROR;
            }
            raised[id] = true;
        }
    }
    if (!launch) {
        return SWARM_MMQ_OK;
    }
    if (fixup_floats < need) {
        return SWARM_MMQ_FIXUP_TOO_SMALL;
    }
    if (config.stream_k && int64_t(ntiles_dst) * blocks_per_ne00_fd.z >= (int64_t(1) << 30)) {
        return SWARM_MMQ_UNSUPPORTED; // `kbc` would overflow (llama.cpp asserts this)
    }

    const uint3 ntx_fd           = init_fastdiv_values(ntx);
    const uint3 nchannels_y_fd   = init_fastdiv_values(args.nchannels_y);
    const uint3 nsamples_y_fd    = init_fastdiv_values(args.nsamples_y);
    const uint3 channel_ratio_fd = init_fastdiv_values(args.nchannels_y / args.nchannels_x);
    const uint3 sample_ratio_fd  = init_fastdiv_values(args.nsamples_y / args.nsamples_x);

    if (!config.stream_k) {
        const dim3 block_nums_xy_tiling(nty, ntx, ntzw);
        mul_mat_q<type, J, fallback><<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>
            (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, nullptr, args.y_scale,
             blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
             channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
             sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
             ntx_fd);
        return cudaGetLastError() == cudaSuccess ? SWARM_MMQ_OK : SWARM_MMQ_CUDA_ERROR;
    }

    const dim3 block_nums_stream_k(nblocks_stream_k, 1, 1);
    mul_mat_q<type, J, fallback><<<block_nums_stream_k, block_dims, nbytes_shared, stream>>>
        (args.x, args.y, args.ids_dst, args.expert_bounds, args.dst, fixup_needed ? fixup : nullptr, args.y_scale,
         blocks_per_ne00_fd, args.nrows_x, args.ncols_dst, args.stride_row_x, args.ncols_y, args.nrows_dst,
         channel_ratio_fd, nchannels_y_fd, args.stride_channel_x, args.stride_channel_y, args.stride_channel_dst,
         sample_ratio_fd, nsamples_y_fd, args.stride_sample_x, args.stride_sample_y, args.stride_sample_dst,
         ntx_fd);
    if (cudaGetLastError() != cudaSuccess) {
        return SWARM_MMQ_CUDA_ERROR;
    }
    if (!fixup_needed) {
        return SWARM_MMQ_OK;
    }
    const dim3 block_nums_fixup(nblocks_stream_k, config.I / dev.warp_size, 1);
    const dim3 block_dims_fixup(block_dims.x, block_dims.y / 2, block_dims.z);
    mul_mat_q_stream_k_fixup<type, J, fallback><<<block_nums_fixup, block_dims_fixup, 0, stream>>>
        (args.ids_dst, args.expert_bounds, args.dst, fixup, blocks_per_ne00_fd, args.nrows_x, args.ncols_dst,
         args.nrows_dst, nchannels_y_fd, args.stride_channel_dst, nsamples_y_fd, args.stride_sample_dst,
         ntx_fd);
    return cudaGetLastError() == cudaSuccess ? SWARM_MMQ_OK : SWARM_MMQ_CUDA_ERROR;
}

// `mul_mat_q_switch_J`: the tile width with the fewest tiles across the
// activation's rows, among those this card's shared memory holds.
template <ggml_type type, bool fallback>
static swarm_mmq_status swarm_mmq_switch_J(
        const swarm_mmq_device & dev, const mmq_args & args, float * fixup, size_t fixup_floats,
        size_t & need, bool launch, cudaStream_t stream) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, dev.cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
        if (mmq_get_nbytes_shared(config, dev.cc) > dev.smpbo) {
            continue;
        }
        const int ntiles_x = (args.ncols_opt + config.J - 1) / config.J;
        if (ntiles_x < ntiles_J_best) {
            J_best        = J;
            ntiles_J_best = ntiles_x;
        }
    }
#define SWARM_MMQ_J(J_) case J_: return swarm_mmq_launch<type, J_, fallback>(dev, args, fixup, fixup_floats, need, launch, stream);
    switch (J_best) {
        SWARM_MMQ_J(8)  SWARM_MMQ_J(16) SWARM_MMQ_J(24)  SWARM_MMQ_J(32)
        SWARM_MMQ_J(40) SWARM_MMQ_J(48) SWARM_MMQ_J(56)  SWARM_MMQ_J(64)
        SWARM_MMQ_J(72) SWARM_MMQ_J(80) SWARM_MMQ_J(88)  SWARM_MMQ_J(96)
        SWARM_MMQ_J(104) SWARM_MMQ_J(112) SWARM_MMQ_J(120) SWARM_MMQ_J(128)
        default: return SWARM_MMQ_NO_TILE;
    }
#undef SWARM_MMQ_J
}

// `mul_mat_q_case` for one type: the bounds-checked kernel only where the
// rows of x do not fill whole 128-row tiles.
template <ggml_type type>
swarm_mmq_status swarm_mmq_case(
        const swarm_mmq_device & dev, const swarm_mmq_dense & p, float * fixup, size_t fixup_floats,
        size_t & need, bool launch, cudaStream_t stream) {
    const mmq_args args = swarm_mmq_args(p, type);
    if (p.nrows_x % 128 == 0) {
        return swarm_mmq_switch_J<type, false>(dev, args, fixup, fixup_floats, need, launch, stream);
    }
    return swarm_mmq_switch_J<type, true>(dev, args, fixup, fixup_floats, need, launch, stream);
}

#define SWARM_MMQ_DECL(type)                                                                       \
    template swarm_mmq_status swarm_mmq_case<type>(                                                \
        const swarm_mmq_device &, const swarm_mmq_dense &, float *, size_t, size_t &, bool, cudaStream_t)

extern SWARM_MMQ_DECL(GGML_TYPE_Q4_0);
extern SWARM_MMQ_DECL(GGML_TYPE_Q4_1);
extern SWARM_MMQ_DECL(GGML_TYPE_Q5_0);
extern SWARM_MMQ_DECL(GGML_TYPE_Q5_1);
extern SWARM_MMQ_DECL(GGML_TYPE_Q8_0);
extern SWARM_MMQ_DECL(GGML_TYPE_Q2_K);
extern SWARM_MMQ_DECL(GGML_TYPE_Q3_K);
extern SWARM_MMQ_DECL(GGML_TYPE_Q4_K);
extern SWARM_MMQ_DECL(GGML_TYPE_Q5_K);
extern SWARM_MMQ_DECL(GGML_TYPE_Q6_K);

} // namespace swarm_ggml
