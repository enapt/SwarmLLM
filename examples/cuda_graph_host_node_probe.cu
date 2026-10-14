// Can a decode step whose experts live in RAM go to the card as CUDA graphs —
// the processor's part a HOST NODE inside the graph — and what does that buy
// over submitting it op by op, on THIS machine (FUTURE_WORK #138 (3))?
//
// WHY THIS EXISTS
// ---------------
// Qwen3-30B-A3B with 25 layers' experts in RAM decodes at ~40-44 ms a token
// here against llama.cpp's ~31 ms at the same placement. The expert products
// are level (19.2 ms vs ~20); the gap is the card's part, ~15 ms of HOST time
// submitting ~20 kernels a layer one by one, because such a step is never
// captured. Capturing it in groups cut at each processor section was tried
// and gained nothing (`pause_capture.patch`, 2026-10-14): every group was
// recorded while the card sat idle at the pause.
//
// KTransformers (kt-kernel/cpu_backend/cpuinfer.h, operators/experts.py) keeps
// the processor's part INSIDE the graph: the router's input, ids and weights
// are copied into FIXED pinned host buffers, `cudaLaunchHostFunc` hands the
// experts to a CPU task queue, a second host function waits for them, and the
// output is copied back from a fixed pinned buffer. Recorded that way, a step
// needs no pause, so recording overlaps the card's run as on a dense model.
// Whether that pays depends on what a host node's round trip costs under
// WSL2's GPU channel — measured here before any of it is built.
//
// RESULT (2026-10-14, RTX 3070 Laptop, WSL2, three rounds): 0.29-0.85 ms per
// host node, so 7-21 ms a token at 25 RAM layers, against the ~15 ms of
// submission a capture saves. Not built (FUTURE_WORK #138 (3)).
//
// One synthetic token = LAYERS layers of LAUNCHES tiny kernels, each writing a
// buffer cudaMallocAsync'd just before it and freed after (candle's pattern),
// one scalar argument changing every token. The first RAM_LAYERS layers also
// do the processor's part: an 8 KB copy to the host, WORK_US of host work, an
// 8 KB copy back.
//
// ARMS (one created stream):
//   direct    today: ops one by one; at a processor part, copy to pinned,
//             synchronize, work on this thread, copy back
//   hostnode  capture in groups of GROUP layers, cudaGraphExecUpdate each
//             group's exec, launch it, record the next while it runs; the
//             processor part is D2H node -> host node (does the work) -> H2D
//   rt        a graph of RAM_LAYERS x (one kernel, D2H, empty host node,
//             H2D), replayed: what one host node's round trip costs
//
// The host work spins on the callback thread; a real one hands off to a
// thread pool and waits, a few µs more. Build and run (no repo build needed):
//   nvcc -O2 -arch=sm_86 examples/cuda_graph_host_node_probe.cu -o /tmp/host_node
//   /tmp/host_node [tokens=60] [work_us=770] [launches=20] [layers=48] [ram_layers=25] [group=2]
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e)); exit(1); } } while (0)

static const int HIDDEN = 2048;  // floats in the residual crossing to the host

__global__ void touch(float* out, const float* in, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (in ? in[i] : 0.0f) + (float)v;
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static void spin_us(int us) {
  auto t = Clock::now();
  while (std::chrono::duration<double, std::micro>(Clock::now() - t).count() < us) {}
}

struct HostWork { float* in; float* out; int work_us; };

static void CUDART_CB do_host_work(void* p) {
  HostWork* w = (HostWork*)p;
  spin_us(w->work_us);
  // Read what came down, write what goes up: the buffers are the contract.
  for (int i = 0; i < HIDDEN; i += 64) w->out[i] = w->in[i] * 0.5f;
}

static void CUDART_CB empty_host_fn(void*) {}

struct Params { int tokens, work_us, launches, layers, ram_layers, group; };

// The card part of one layer: `launches` alloc + kernel + free, chained on x.
static void card_part(cudaStream_t s, float* x, int launches, int v) {
  for (int k = 0; k < launches; ++k) {
    float* t = nullptr;
    CK(cudaMallocAsync((void**)&t, HIDDEN * sizeof(float), s));
    touch<<<(HIDDEN + 255) / 256, 256, 0, s>>>(t, x, HIDDEN, v + k);
    touch<<<(HIDDEN + 255) / 256, 256, 0, s>>>(x, t, HIDDEN, 0);
    CK(cudaFreeAsync(t, s));
  }
}

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  Params p{60, 770, 20, 48, 25, 2};
  if (argc > 1) p.tokens = atoi(argv[1]);
  if (argc > 2) p.work_us = atoi(argv[2]);
  if (argc > 3) p.launches = atoi(argv[3]);
  if (argc > 4) p.layers = atoi(argv[4]);
  if (argc > 5) p.ram_layers = atoi(argv[5]);
  if (argc > 6) p.group = atoi(argv[6]);
  printf("tokens=%d work_us=%d launches/layer=%d (x2 kernels) layers=%d ram_layers=%d group=%d\n",
         p.tokens, p.work_us, p.launches, p.layers, p.ram_layers, p.group);

  cudaStream_t s;
  CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  float *x, *pin_in, *pin_out;
  CK(cudaMalloc((void**)&x, HIDDEN * sizeof(float)));
  CK(cudaMemset(x, 0, HIDDEN * sizeof(float)));
  // NOT write-combined: the host READS what comes down.
  CK(cudaHostAlloc((void**)&pin_in, HIDDEN * sizeof(float), cudaHostAllocDefault));
  CK(cudaHostAlloc((void**)&pin_out, HIDDEN * sizeof(float), cudaHostAllocDefault));
  memset(pin_out, 0, HIDDEN * sizeof(float));
  HostWork work{pin_in, pin_out, p.work_us};

  // ---- direct ----
  std::vector<double> direct;
  for (int tok = 0; tok < p.tokens + 3; ++tok) {
    auto t = Clock::now();
    for (int l = 0; l < p.layers; ++l) {
      card_part(s, x, p.launches, tok);
      if (l < p.ram_layers) {
        CK(cudaMemcpyAsync(pin_in, x, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        do_host_work(&work);
        CK(cudaMemcpyAsync(x, pin_out, HIDDEN * sizeof(float), cudaMemcpyHostToDevice, s));
      }
    }
    CK(cudaStreamSynchronize(s));
    if (tok >= 3) direct.push_back(ms_since(t));
  }

  // ---- hostnode: grouped capture, each group updated in place ----
  int groups = (p.layers + p.group - 1) / p.group;
  std::vector<cudaGraphExec_t> execs(groups, nullptr);
  std::vector<double> hostnode;
  int refused = 0;
  double recording_ms = 0;
  for (int tok = 0; tok < p.tokens + 3; ++tok) {
    auto t = Clock::now();
    for (int g = 0; g < groups; ++g) {
      auto r = Clock::now();
      CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      for (int l = g * p.group; l < std::min(p.layers, (g + 1) * p.group); ++l) {
        card_part(s, x, p.launches, tok);
        if (l < p.ram_layers) {
          CK(cudaMemcpyAsync(pin_in, x, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost, s));
          CK(cudaLaunchHostFunc(s, do_host_work, &work));
          CK(cudaMemcpyAsync(x, pin_out, HIDDEN * sizeof(float), cudaMemcpyHostToDevice, s));
        }
      }
      cudaGraph_t graph;
      CK(cudaStreamEndCapture(s, &graph));
      bool updated = false;
      if (execs[g]) {
        cudaGraphExecUpdateResultInfo info;
        updated = cudaGraphExecUpdate(execs[g], graph, &info) == cudaSuccess;
        if (!updated) { ++refused; cudaGetLastError(); CK(cudaGraphExecDestroy(execs[g])); execs[g] = nullptr; }
      }
      if (!execs[g]) CK(cudaGraphInstantiate(&execs[g], graph, 0));
      CK(cudaGraphDestroy(graph));
      if (tok >= 3) recording_ms += ms_since(r);
      CK(cudaGraphLaunch(execs[g], s));
    }
    CK(cudaStreamSynchronize(s));
    if (tok >= 3) hostnode.push_back(ms_since(t));
  }

  // ---- rt: one host node's round trip, replayed ----
  cudaGraph_t rtg;
  cudaGraphExec_t rtx;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int l = 0; l < p.ram_layers; ++l) {
    touch<<<(HIDDEN + 255) / 256, 256, 0, s>>>(x, nullptr, HIDDEN, l);
    CK(cudaMemcpyAsync(pin_in, x, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaLaunchHostFunc(s, empty_host_fn, nullptr));
    CK(cudaMemcpyAsync(x, pin_out, HIDDEN * sizeof(float), cudaMemcpyHostToDevice, s));
  }
  CK(cudaStreamEndCapture(s, &rtg));
  CK(cudaGraphInstantiate(&rtx, rtg, 0));
  cudaGraph_t nog;  // the same without host nodes: what the copies cost alone
  cudaGraphExec_t nox;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int l = 0; l < p.ram_layers; ++l) {
    touch<<<(HIDDEN + 255) / 256, 256, 0, s>>>(x, nullptr, HIDDEN, l);
    CK(cudaMemcpyAsync(pin_in, x, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(x, pin_out, HIDDEN * sizeof(float), cudaMemcpyHostToDevice, s));
  }
  CK(cudaStreamEndCapture(s, &nog));
  CK(cudaGraphInstantiate(&nox, nog, 0));
  std::vector<double> rt, nohost;
  for (int i = 0; i < p.tokens + 3; ++i) {
    auto t = Clock::now();
    CK(cudaGraphLaunch(rtx, s));
    CK(cudaStreamSynchronize(s));
    if (i >= 3) rt.push_back(ms_since(t));
    t = Clock::now();
    CK(cudaGraphLaunch(nox, s));
    CK(cudaStreamSynchronize(s));
    if (i >= 3) nohost.push_back(ms_since(t));
  }

  double floor_ms = p.ram_layers * p.work_us / 1000.0;
  printf("host work alone (the floor no arm beats): %.2f ms a token\n", floor_ms);
  printf("direct   : median %.2f ms, min %.2f ms a token\n", median(direct),
         *std::min_element(direct.begin(), direct.end()));
  printf("hostnode : median %.2f ms, min %.2f ms a token  (recording+update %.2f ms a token, %d updates refused)\n",
         median(hostnode), *std::min_element(hostnode.begin(), hostnode.end()),
         recording_ms / p.tokens, refused);
  printf("rt       : %d x (kernel, D2H, empty host node, H2D) median %.3f ms; without the host node %.3f ms"
         " -> %.1f us a host node\n", p.ram_layers, median(rt), median(nohost),
         (median(rt) - median(nohost)) * 1000.0 / p.ram_layers);
  printf("check    : pin_out[0]=%g (written by the host work)\n", pin_out[0]);
  return 0;
}
