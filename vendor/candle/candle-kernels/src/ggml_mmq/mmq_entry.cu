// SwarmLLM: the C entry points of llama.cpp's MMQ, called from the vendored
// candle-core (`quantized/cuda.rs`, declared in candle-kernels' `ffi.rs`).
// Why this kernel, and what was replaced: `swarm_mmq.cuh`.
#include "swarm_mmq.cuh"

namespace swarm_ggml {

#include "ggml-cuda/quantize.cuh"
// llama.cpp's activation quantizers, verbatim (renamed so no build compiles
// it on its own: every kernel here is reached through this file).
#include "ggml-cuda/quantize.cu.inc"

// ggml's fatal-error hooks, which the copied headers call and which ggml.c
// and ggml-cuda.cu define (`ggml_abort` is `swarm_ggml_abort` here, see
// swarm_mmq.cuh). The launchers here check what they can first and
// return a status, so these are reached only by an assertion inside llama.cpp
// code that a valid call cannot trip.
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    fprintf(stderr, "%s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    fprintf(stderr, "CUDA error: %s\n  in %s at %s:%d\n  %s\n", msg, func, file, line, stmt);
    abort();
}

// ggml-cuda.cu's, verbatim: the quantizers' launch helper asks it which
// device a kernel runs on (`ggml_cuda_kernel_can_use_pdl`).
int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

// An empty kernel built with the rest of this library: where the driver has
// no image of it for the current card, it has none of the others either.
__global__ void swarm_mmq_probe() {}

swarm_mmq_status swarm_mmq_current_device(swarm_mmq_device & out) {
    static std::mutex       mu;
    static bool             known[GGML_CUDA_MAX_DEVICES] = { false };
    static swarm_mmq_device cache[GGML_CUDA_MAX_DEVICES];

    int id = 0;
    if (cudaGetDevice(&id) != cudaSuccess || id < 0 || id >= GGML_CUDA_MAX_DEVICES) {
        return SWARM_MMQ_CUDA_ERROR;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (!known[id]) {
        int major = 0, minor = 0, nsm = 0, smpbo = 0, warp = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, id) != cudaSuccess ||
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, id) != cudaSuccess ||
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, id) != cudaSuccess ||
            cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, id) != cudaSuccess ||
            cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, id) != cudaSuccess) {
            return SWARM_MMQ_CUDA_ERROR;
        }
        cache[id] = swarm_mmq_device{ 100*major + 10*minor, nsm, size_t(smpbo), warp };
        known[id] = true;
    }
    out = cache[id];
    return SWARM_MMQ_OK;
}

} // namespace swarm_ggml

using namespace swarm_ggml;

// The types this build instantiates (the ones candle's own MMQ covered).
static int swarm_mmq_block_size(const int type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return 32;
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return 256;
        default:
            return 0;
    }
}

// 1 where this kernel runs on the current device: int8 tensor cores
// (llama.cpp's `turing_mma_available`, which is also when llama.cpp itself
// takes MMQ for every batch), the 48 KiB of shared memory its tiles need, and
// an image of this library the driver can run here. The last is not implied
// by the first: `turing_mma_available` reads the architectures this file was
// compiled for (`__CUDA_ARCH_LIST__`), so it already says no to a card OLDER
// than the build, but yes to a newer one whose machine code it lacks (a 12.0
// card, a library built for 80). That card runs the library only from its
// PTX; without one, the first launch fails and the old kernels take over —
// this asks first instead.
extern "C" int swarm_mmq_available() {
    swarm_mmq_device dev;
    if (swarm_mmq_current_device(dev) != SWARM_MMQ_OK) {
        return 0;
    }
    if (!turing_mma_available(dev.cc) || dev.smpbo < 48 * 1024) {
        return 0;
    }
    cudaFuncAttributes attr;
    if (cudaFuncGetAttributes(&attr, swarm_mmq_probe) != cudaSuccess) {
        (void) cudaGetLastError(); // not sticky: leave nothing for a later check to read
        return 0;
    }
    return 1;
}

// Each launch here is judged by the thread's last runtime error afterwards.
// An error an earlier, unrelated runtime call left behind would be read as the
// launch's own — and turn this kernel off for the process — so it is cleared
// before anything here launches.
static void swarm_mmq_clear_stale_error() {
    (void) cudaGetLastError();
}

// Which `block_q8_1_mmq` scale layout a weight type reads its activation in
// (0 = D4, 1 = DS4, 2 = D2S6), or -1 for a type not covered. Two weights
// whose layouts agree can share one quantized activation.
extern "C" int swarm_mmq_q8_1_layout(const int type) {
    if (swarm_mmq_block_size(type) == 0) {
        return -1;
    }
    return int(mmq_get_q8_1_ds_layout(ggml_type(type)));
}

// Bytes of the quantized activation for `ncols_y` rows of `ncols_x` values,
// with the tail llama.cpp allocates for a tile reading past the last row
// (`ggml_cuda_mmq_get_J_max` rows; the widest tile, 128, is always enough).
extern "C" size_t swarm_mmq_q8_1_bytes(const int64_t ncols_x, const int64_t ncols_y) {
    const int64_t padded = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    return size_t(ncols_y * padded) * sizeof(block_q8_1) / QK8_1 + 128 * sizeof(block_q8_1_mmq);
}

// `ncols_y` rows of `ncols_x` floats (`stride_y` floats apart) into `vy`, in
// the layout a weight of `type` reads.
extern "C" int swarm_mmq_quantize(
        const float * y, void * vy, const int type, const int64_t ncols_x, const int64_t ncols_y,
        const int64_t stride_y, const int64_t stream) {
    if (swarm_mmq_block_size(type) == 0 || ncols_x % 4 != 0 || ncols_y <= 0) {
        return SWARM_MMQ_UNSUPPORTED;
    }
    const int64_t padded = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    swarm_mmq_clear_stale_error();
    quantize_mmq_q8_1_cuda(y, nullptr, vy, ggml_type(type), ncols_x, stride_y, stride_y*ncols_y, stride_y*ncols_y,
                           padded, ncols_y, 1, 1, (cudaStream_t) stream);
    return cudaGetLastError() == cudaSuccess ? SWARM_MMQ_OK : SWARM_MMQ_CUDA_ERROR;
}

// dst[ncols_y][nrows_x] = vy · xᵀ. With `launch == 0` nothing runs: `*need`
// is set to the floats of fixup buffer the launch will ask for (0 = none),
// so the caller can allocate it on its own pool and stream. With
// `launch != 0` the product is enqueued on `stream`.
extern "C" int swarm_mmq_matmul(
        const int type, const void * x, const void * vy, float * dst, float * fixup, const size_t fixup_floats,
        size_t * need, const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y, const int launch,
        const int64_t stream) {
    const int qk = swarm_mmq_block_size(type);
    if (qk == 0 || ncols_x % qk != 0 || nrows_x <= 0 || ncols_y <= 0 || need == nullptr) {
        return SWARM_MMQ_UNSUPPORTED;
    }
    swarm_mmq_device dev;
    const swarm_mmq_status got = swarm_mmq_current_device(dev);
    if (got != SWARM_MMQ_OK) {
        return got;
    }
    const swarm_mmq_dense p = { (const char *) x, (const int *) vy, dst, ncols_x, nrows_x, ncols_y, ncols_x / qk };
    size_t needed = 0;
    swarm_mmq_status status = SWARM_MMQ_UNSUPPORTED;
    const cudaStream_t s = (cudaStream_t) stream;
    const bool go = launch != 0;
    if (go) {
        swarm_mmq_clear_stale_error();
    }
    switch (type) {
        case GGML_TYPE_Q4_0: status = swarm_mmq_case<GGML_TYPE_Q4_0>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q4_1: status = swarm_mmq_case<GGML_TYPE_Q4_1>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q5_0: status = swarm_mmq_case<GGML_TYPE_Q5_0>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q5_1: status = swarm_mmq_case<GGML_TYPE_Q5_1>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q8_0: status = swarm_mmq_case<GGML_TYPE_Q8_0>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q2_K: status = swarm_mmq_case<GGML_TYPE_Q2_K>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q3_K: status = swarm_mmq_case<GGML_TYPE_Q3_K>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q4_K: status = swarm_mmq_case<GGML_TYPE_Q4_K>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q5_K: status = swarm_mmq_case<GGML_TYPE_Q5_K>(dev, p, fixup, fixup_floats, needed, go, s); break;
        case GGML_TYPE_Q6_K: status = swarm_mmq_case<GGML_TYPE_Q6_K>(dev, p, fixup, fixup_floats, needed, go, s); break;
        default: break;
    }
    *need = needed;
    return status;
}
