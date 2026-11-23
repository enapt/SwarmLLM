// What would a CUDA graph save per decoded token on THIS machine — and does
// re-capturing every token (llama.cpp's way) keep the saving?
//
// WHY THIS EXISTS
// ---------------
// `docs/plans/local_decode_submissions.md` measured decode as bound by
// SUBMISSIONS: ~1,085 per token on a 22-layer model (625 kernel launches, 657
// cudaMallocAsync + 656 cudaFreeAsync), ~10 µs each under WSL2. A graph
// replays a whole sequence with one submission. The plan's open question is
// how to graph a step whose kernel PARAMETERS change every token (the KV
// length): capture once per length class (HuggingFace grout) or re-capture
// every token and `cudaGraphExecUpdate` the instantiated graph (llama.cpp).
// Re-capturing records every launch again — cheap if recording stays in user
// space, useless if WSL2 marshals it to the host like a real launch. That is
// measured here, before any of the four preconditions in § Stage 4b is built.
// `cuda_graph_probe.cu` established capability; this measures cost.
//
// One synthetic "token" = LAUNCHES tiny kernels, each writing a buffer
// allocated with cudaMallocAsync just before it and freed after — candle's
// pattern — with one scalar argument that changes every token.
//
// ARMS (all on one created stream, the legacy stream cannot capture):
//   direct   launch + alloc/free per op, one sync per token     (today)
//   update   capture every token, cudaGraphExecUpdate the exec, launch
//            (instantiate afresh when the update is refused)    (llama.cpp)
//   replay   capture ONCE, replay the same exec every token      (upper bound)
//   arena    as update, but every op writes a buffer made once OUTSIDE the
//            capture (a planned arena, ggml-alloc's way): the graph holds the
//            kernels only, no memory nodes                  (2026-11-23, #228 (1))
//
// MEASURED 2026-11-23 (this RTX 3060 laptop card under WSL2, 200 tokens, two
// runs each): 1000 launches — update 3.68-4.00 ms/token (capture 1.53-1.58,
// update 0.50-0.57), arena 1.61-1.64 (capture 0.44-0.46, update 0.19), replay
// 0.92-0.95; 85 launches — update 0.28, arena 0.18. The memory nodes are ~2 µs
// a pair of recording, update and the card's own run HERE — but a real
// Qwen3.5-0.8B level with its ~900 pairs served from a pool ran no faster
// (FUTURE_WORK § Decided, #496): read this probe's rates as its own.
//
// WHAT IT CANNOT TELL YOU: anything about OUR forward pass (captured allocs
// must be freed inside the capture, no host sync inside it — § Stage 4b), or
// the kernels' own run time (these are near-empty on purpose: the question is
// the submission cost). Build and run (no repo build needed):
//   nvcc -O2 -arch=sm_86 examples/cuda_graph_cost_probe.cu -o /tmp/graph_cost
//   /tmp/graph_cost [launches=600] [tokens=200]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %d (%s) at %s:%d\n", (int)e, cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void tiny(float* out, int n, int token) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(token + i);
}

static void one_token(cudaStream_t s, int launches, int token) {
    for (int k = 0; k < launches; k++) {
        float* buf = nullptr;
        CK(cudaMallocAsync((void**)&buf, 4096 * sizeof(float), s));
        tiny<<<16, 256, 0, s>>>(buf, 4096, token);
        CK(cudaFreeAsync(buf, s));
    }
}

static void one_token_arena(cudaStream_t s, int launches, int token, float** bufs) {
    for (int k = 0; k < launches; k++) {
        tiny<<<16, 256, 0, s>>>(bufs[k], 4096, token);
    }
}

static double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main(int argc, char** argv) {
    int launches = argc > 1 ? atoi(argv[1]) : 600;
    int tokens = argc > 2 ? atoi(argv[2]) : 200;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // Keep freed memory in the pool, as the node does since #146 — otherwise
    // every sync would hand it back and the direct arm would pay that too.
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, 0));
    unsigned long long threshold = ~0ULL;
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));

    // Warm up the context, the module and the pool.
    for (int t = 0; t < 5; t++) one_token(s, launches, t);
    CK(cudaStreamSynchronize(s));

    // direct
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < tokens; t++) { one_token(s, launches, t); CK(cudaStreamSynchronize(s)); }
    double direct = ms_since(t0) / tokens;

    // update: re-capture every token, update the instantiated graph
    cudaGraphExec_t exec = nullptr;
    int updated = 0, reinstantiated = 0;
    double capture_ms = 0, update_ms = 0;
    t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < tokens; t++) {
        auto c0 = std::chrono::steady_clock::now();
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        one_token(s, launches, t);
        CK(cudaStreamEndCapture(s, &g));
        capture_ms += ms_since(c0);
        auto u0 = std::chrono::steady_clock::now();
        if (exec) {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(exec, g, &info) == cudaSuccess) {
                updated++;
            } else {
                cudaGetLastError();
                CK(cudaGraphExecDestroy(exec));
                CK(cudaGraphInstantiate(&exec, g, 0));
                reinstantiated++;
            }
        } else {
            CK(cudaGraphInstantiate(&exec, g, 0));
            reinstantiated++;
        }
        update_ms += ms_since(u0);
        CK(cudaGraphLaunch(exec, s));
        CK(cudaStreamSynchronize(s));
        CK(cudaGraphDestroy(g));
    }
    double update = ms_since(t0) / tokens;
    CK(cudaGraphExecDestroy(exec));

    // replay: capture once
    cudaGraph_t g1;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    one_token(s, launches, 0);
    CK(cudaStreamEndCapture(s, &g1));
    cudaGraphExec_t e1;
    CK(cudaGraphInstantiate(&e1, g1, 0));
    CK(cudaGraphLaunch(e1, s));
    CK(cudaStreamSynchronize(s));
    t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < tokens; t++) { CK(cudaGraphLaunch(e1, s)); CK(cudaStreamSynchronize(s)); }
    double replay = ms_since(t0) / tokens;

    CK(cudaGraphExecDestroy(e1));
    CK(cudaGraphDestroy(g1));

    // arena: re-capture every token as `update`, the buffers made once outside
    float** bufs = (float**)malloc(sizeof(float*) * launches);
    for (int k = 0; k < launches; k++) CK(cudaMalloc((void**)&bufs[k], 4096 * sizeof(float)));
    cudaGraphExec_t ea = nullptr;
    int a_updated = 0, a_inst = 0;
    double a_capture_ms = 0, a_update_ms = 0, a_launch_ms = 0;
    t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < tokens; t++) {
        auto c0 = std::chrono::steady_clock::now();
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        one_token_arena(s, launches, t, bufs);
        CK(cudaStreamEndCapture(s, &g));
        a_capture_ms += ms_since(c0);
        auto u0 = std::chrono::steady_clock::now();
        if (ea) {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(ea, g, &info) == cudaSuccess) {
                a_updated++;
            } else {
                cudaGetLastError();
                CK(cudaGraphExecDestroy(ea));
                CK(cudaGraphInstantiate(&ea, g, 0));
                a_inst++;
            }
        } else {
            CK(cudaGraphInstantiate(&ea, g, 0));
            a_inst++;
        }
        a_update_ms += ms_since(u0);
        auto l0 = std::chrono::steady_clock::now();
        CK(cudaGraphLaunch(ea, s));
        a_launch_ms += ms_since(l0);
        CK(cudaStreamSynchronize(s));
        CK(cudaGraphDestroy(g));
    }
    double arena = ms_since(t0) / tokens;

    printf("launches/token=%d tokens=%d\n", launches, tokens);
    printf("direct : %.3f ms/token (%d launches + %d alloc/free pairs, one sync)\n", direct, launches, launches);
    printf("update : %.3f ms/token (capture %.3f, update/instantiate %.3f; %d updated, %d instantiated)\n",
           update, capture_ms / tokens, update_ms / tokens, updated, reinstantiated);
    printf("replay : %.3f ms/token (upper bound: one graph launch)\n", replay);
    printf("arena  : %.3f ms/token (capture %.3f, update %.3f, launch call %.3f; %d updated, %d instantiated)\n",
           arena, a_capture_ms / tokens, a_update_ms / tokens, a_launch_ms / tokens, a_updated, a_inst);
    return 0;
}
