// Can this machine capture a CUDA graph, and does capturing an allocation work?
//
// WHY THIS EXISTS
// ---------------
// `docs/plans/local_decode_submissions.md` § Stage 4 rests on two claims that
// arrived from NVIDIA's documentation and nothing else. Both gate real work,
// so both are checked here against the driver actually installed:
//
//   A. Capture is IMPOSSIBLE on the legacy NULL stream — which is the stream
//      candle's `BackendDevice::new` takes (`context.default_stream()` is
//      `cu_stream: null_mut()`). If A holds, Stage 4 cannot begin until every
//      `CudaDevice` moves to an explicitly created stream.
//
//   B. `cudaMallocAsync` captured into a graph becomes a graph memory node
//      whose address is fixed across replays. If B holds, candle allocating
//      every op output fresh is NOT an obstacle to graph capture, and
//      Stage 3 (a stable-buffer arena) is not a prerequisite for Stage 4 —
//      which is a multi-day refactor either saved or justified by this one
//      answer.
//
// A doubles as a null control. If capture on the legacy stream SUCCEEDS, the
// blocker recorded in the plan is wrong and Stage 4 is cheaper than stated. A
// probe that can only confirm what it was written to confirm is worth nothing:
// four of five repo guards checked in 2026-08 could not see the defect they
// existed to catch, and had been green for months (gotcha #413).
//
// WHAT IT CANNOT TELL YOU
// -----------------------
//   * Nothing about whether OUR decode step can be captured. A real forward
//     pass must also free inside the same capture everything it allocates
//     there, and must contain no host synchronisation — see § Ordering item 3b.
//     This establishes the platform, not the program.
//   * Nothing about speed. It measures capability, not time.
//
// ⚠ **Worth running on a NON-WSL2 NVIDIA box.** Everything in the decode plan
// was measured under WSL2 on one machine, which that document names as its
// single biggest unknown. This probe needs no repo build — just nvcc — so it
// is the cheapest thing to hand to anyone with a different card.
//
// Build and run:
//   nvcc -O2 -o /tmp/cuda_graph_probe examples/cuda_graph_probe.cu
//   /tmp/cuda_graph_probe
// (add `-arch=sm_XX` if nvcc cannot detect the card; exit 0 = both hold)

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("  FAIL %s -> %s\n", #x, cudaGetErrorString(e)); return false; } } while (0)

__global__ void fill(float *p, int n, float v) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        p[i] = v * (i + 1);
}

__global__ void scale_into(const float *src, float *dst, int n, float k) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = src[i] * k;
}

// Claim A — capture on the LEGACY null stream.
static bool probe_legacy_stream_capture() {
    printf("A. capture on the legacy NULL stream (cudaStreamLegacy):\n");
    cudaError_t e = cudaStreamBeginCapture(cudaStreamLegacy, cudaStreamCaptureModeGlobal);
    if (e != cudaSuccess) {
        printf("  REFUSED: %s (cudaError %d)\n", cudaGetErrorString(e), (int)e);
        printf("  -> claim A HOLDS: the stream candle uses today cannot be captured.\n");
        cudaGetLastError();  // clear it; the refusal is the result, not a fault
        return true;
    }
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(cudaStreamLegacy, &g);
    if (g) cudaGraphDestroy(g);
    printf("  ACCEPTED -> claim A IS WRONG; the plan overstates the blocker.\n");
    return true;
}

// Claim B — an allocation captured into the graph, replayed.
static bool probe_graph_alloc_stability() {
    const int N = 4096;
    printf("B. cudaMallocAsync captured into a graph, replayed:\n");

    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    // Persistent output allocated OUTSIDE the capture — what a decode step's
    // KV cache and logits buffer look like.
    float *out = nullptr;
    CHECK(cudaMalloc(&out, N * sizeof(float)));
    CHECK(cudaMemset(out, 0, N * sizeof(float)));

    void *captured_ptr = nullptr;
    CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    {
        // Allocated INSIDE the capture, exactly as candle allocates every op
        // output inside a forward pass.
        float *tmp = nullptr;
        CHECK(cudaMallocAsync((void **)&tmp, N * sizeof(float), s));
        captured_ptr = tmp;
        fill<<<32, 128, 0, s>>>(tmp, N, 1.0f);
        scale_into<<<32, 128, 0, s>>>(tmp, out, N, 2.0f);
        // Freed inside the SAME capture — the documented requirement.
        CHECK(cudaFreeAsync(tmp, s));
    }
    cudaGraph_t graph = nullptr;
    CHECK(cudaStreamEndCapture(s, &graph));
    printf("  captured ok; pointer handed out during capture: %p\n", captured_ptr);

    cudaGraphExec_t exec = nullptr;
    CHECK(cudaGraphInstantiate(&exec, graph, 0));

    float host[4];
    bool all_ok = true;
    for (int rep = 1; rep <= 3; ++rep) {
        CHECK(cudaMemsetAsync(out, 0, N * sizeof(float), s));
        CHECK(cudaGraphLaunch(exec, s));
        CHECK(cudaStreamSynchronize(s));
        CHECK(cudaMemcpy(host, out, sizeof(host), cudaMemcpyDeviceToHost));
        // fill writes v*(i+1) with v=1; scale_into multiplies by 2 -> 2*(i+1).
        // Zeroing `out` before each replay is what makes a replay that wrote
        // nowhere useful show up as zeros rather than the previous answer.
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && (host[i] == 2.0f * (i + 1));
        printf("  replay %d: out[0..3] = %.1f %.1f %.1f %.1f  %s\n",
               rep, host[0], host[1], host[2], host[3], ok ? "CORRECT" : "WRONG");
        all_ok = all_ok && ok;
    }

    printf(all_ok
        ? "  -> claim B HOLDS: allocations captured into the graph replay correctly.\n"
          "     Stage 3 (stable activation buffers) is NOT implied by graph capture.\n"
        : "  -> claim B FAILS: replays diverge. Stage 3 really is a prerequisite.\n");

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(out);
    cudaStreamDestroy(s);
    return all_ok;
}

int main() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        printf("no CUDA device\n");
        return 2;
    }
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    printf("device: %s  sm_%d%d  runtime %d driver %d\n\n",
           prop.name, prop.major, prop.minor, rt, drv);

    bool a = probe_legacy_stream_capture();
    printf("\n");
    bool b = probe_graph_alloc_stability();
    return (a && b) ? 0 : 1;
}
