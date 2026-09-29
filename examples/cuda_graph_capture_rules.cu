// Two rules a CUDA graph capture of our decode forward depends on, measured on
// THIS machine rather than read off the docs.
//
// WHY THIS EXISTS
// ---------------
// `src/inference/cuda_graph.rs` re-captures every decoded token's forward and
// updates one instantiated graph in place (llama.cpp's approach).
// `cuda_graph_cost_probe.cu` showed that pays on WSL2 when every token has the
// same shape. Two questions were left, and the design turns on both:
//
//   A. Our attention scores GROW by one position a token, so one allocation's
//      size — and its kernel's launch grid — changes every capture. Does
//      `cudaGraphExecUpdate` accept that, or does every token re-instantiate
//      (which would forfeit the saving and force fixed-size buffers)?
//   B. Candle uploads small layouts from temporary host vectors
//      (`clone_htod`). Is a pageable host->device copy REFUSED inside a
//      capture (loud, safe), or captured and replayed from the host address at
//      LAUNCH, after the temporary is gone (silent garbage)?
//
// RESULTS (RTX 3070 Laptop, WSL2, driver 581.x, CUDA 12.x, 2026-09-29):
//   A. accepted: 199/199 updates, none re-instantiated, with every tenth
//      allocation growing each token, and with its grid growing too.
//   B. CAPTURED, NOT REFUSED — and read at launch: a value written to the host
//      buffer after the capture is the one that reaches the card. Hence the
//      always-on copy counter in vendored candle (`htod_copies_so_far`) and the
//      capture that throws itself away when it moved.
//
// Build and run (no repo build needed):
//   nvcc -O2 -arch=sm_86 examples/cuda_graph_capture_rules.cu -o /tmp/capture_rules -lcuda
//   /tmp/capture_rules [launches=1000] [tokens=200]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %d (%s) at line %d\n", (int)e, cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void tiny(float* out, int n, int token) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)(token + i);
}

// mode 0: every op the same size; 1: every tenth allocation grows per token;
// 2: the same, and its kernel's grid grows with it.
static void one_token(cudaStream_t s, int launches, int token, int mode) {
    for (int k = 0; k < launches; k++) {
        size_t n = 4096;
        if (mode >= 1 && k % 10 == 0) n = 4096 + (size_t)token * 37;
        float* buf;
        CK(cudaMallocAsync((void**)&buf, n * sizeof(float), s));
        int blocks = (mode == 2) ? (int)((n + 255) / 256) : 16;
        tiny<<<blocks, 256, 0, s>>>(buf, (int)n, token);
        CK(cudaFreeAsync(buf, s));
    }
}

static double ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

static void question_a(cudaStream_t s, int launches, int tokens) {
    const char* names[] = {"same sizes     ", "growing allocs ", "growing + grid "};
    for (int mode = 0; mode < 3; mode++) {
        cudaGraphExec_t exec = nullptr;
        int updated = 0, instantiated = 0;
        cudaGraphExecUpdateResult first_refusal = cudaGraphExecUpdateSuccess;
        auto t0 = std::chrono::steady_clock::now();
        for (int t = 0; t < tokens; t++) {
            cudaGraph_t g;
            CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
            one_token(s, launches, t, mode);
            CK(cudaStreamEndCapture(s, &g));
            if (exec) {
                cudaGraphExecUpdateResultInfo info;
                if (cudaGraphExecUpdate(exec, g, &info) == cudaSuccess) {
                    updated++;
                } else {
                    if (first_refusal == cudaGraphExecUpdateSuccess) first_refusal = info.result;
                    cudaGetLastError();
                    CK(cudaGraphExecDestroy(exec));
                    CK(cudaGraphInstantiate(&exec, g, 0));
                    instantiated++;
                }
            } else {
                CK(cudaGraphInstantiate(&exec, g, 0));
                instantiated++;
            }
            CK(cudaGraphLaunch(exec, s));
            CK(cudaStreamSynchronize(s));
            CK(cudaGraphDestroy(g));
        }
        printf("A %s: %.3f ms/token, %d updated, %d instantiated, first refusal code %d\n",
               names[mode], ms(t0) / tokens, updated, instantiated, (int)first_refusal);
        CK(cudaGraphExecDestroy(exec));
    }
}

static void question_b(cudaStream_t s) {
    size_t* host = (size_t*)malloc(2 * sizeof(size_t));  // pageable, like a Vec
    host[0] = 111; host[1] = 222;
    size_t* dev;
    CK(cudaMalloc(&dev, 2 * sizeof(size_t)));
    CK(cudaMemset(dev, 0, 2 * sizeof(size_t)));
    CK(cudaDeviceSynchronize());
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    // The driver call cudarc makes for `clone_htod` of a slice.
    CUresult r = cuMemcpyHtoDAsync((CUdeviceptr)dev, host, 2 * sizeof(size_t), (CUstream)s);
    cudaError_t ended = cudaStreamEndCapture(s, &g);
    printf("B pageable cuMemcpyHtoDAsync inside a capture: %d; end capture: %d (%s)\n",
           (int)r, (int)ended, cudaGetErrorString(ended));
    if (ended != cudaSuccess) return;
    host[0] = 999; host[1] = 888;  // written AFTER the capture, BEFORE the launch
    cudaGraphExec_t x;
    CK(cudaGraphInstantiate(&x, g, 0));
    CK(cudaGraphLaunch(x, s));
    CK(cudaStreamSynchronize(s));
    size_t back[2];
    CK(cudaMemcpy(back, dev, sizeof back, cudaMemcpyDeviceToHost));
    printf("B the card holds %zu %zu — 111 222 means read at capture, 999 888 means read at LAUNCH\n",
           back[0], back[1]);
}

int main(int argc, char** argv) {
    int launches = argc > 1 ? atoi(argv[1]) : 1000;
    int tokens = argc > 2 ? atoi(argv[2]) : 200;
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    // Keep freed memory in the pool, as the node does since #146.
    cudaMemPool_t pool;
    CK(cudaDeviceGetDefaultMemPool(&pool, 0));
    unsigned long long threshold = ~0ULL;
    CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    for (int t = 0; t < 5; t++) one_token(s, launches, t, 0);
    CK(cudaStreamSynchronize(s));
    question_a(s, launches, tokens);
    question_b(s);
    return 0;
}
