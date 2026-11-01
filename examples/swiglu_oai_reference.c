// llama.cpp's own `ggml_swiglu_oai` (gpt-oss's activation) on fixed inputs:
// the figure `fast_math::tests::swiglu_oai_is_llama_cpps_expression` asserts.
//
// Build against a llama.cpp checkout's CPU build (no cargo; not an [[example]]):
//
//   R=~/llama.cpp-ref
//   gcc -O2 examples/swiglu_oai_reference.c -I$R/ggml/include -L$R/build/bin \
//     -lggml-base -lggml-cpu -lm -Wl,-rpath,$R/build/bin -o /tmp/swiglu_oai_reference
//   /tmp/swiglu_oai_reference
//
// The inputs are eleven values either side of both clamps, then a xorshift32
// spread the Rust test repeats step for step. It prints an FNV-1a hash of the
// output's bits, computed by the op llama.cpp's gpt-oss graph builds
// (`LLM_FFN_SWIGLU_OAI_MOE`: alpha 1.702, limit 7, gate first).
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml-cpu.h"
#include "ggml.h"

#define N 4104

static uint32_t s = 0x9e3779b9u;
static uint32_t next(void) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

int main(void) {
    static const float g0[11] = {-90.0f, -7.5f, -7.0f, -1.0f, -0.0f, 0.0f, 0.3f, 6.99f, 7.0f, 7.01f, 50.0f};
    static const float u0[11] = {-50.0f, -7.01f, -7.0f, -1.0f, 0.0f, -0.0f, 0.7f, 6.99f, 7.0f, 7.01f, 90.0f};

    struct ggml_init_params ip = {.mem_size = 16 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = false};
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *gate = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, N / 8);
    struct ggml_tensor *up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, N / 8);
    float *g = (float *)gate->data;
    float *u = (float *)up->data;
    memcpy(g, g0, sizeof g0);
    memcpy(u, u0, sizeof u0);
    for (int i = 11; i < N; i++) {
        g[i] = (float)(next() % 40001) / 2000.0f - 10.0f;
        u[i] = (float)(next() % 40001) / 2000.0f - 10.0f;
    }

    struct ggml_tensor *out = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    const uint8_t *p = (const uint8_t *)out->data;
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < N * sizeof(float); i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    const float *o = (const float *)out->data;
    printf("swiglu_oai_fnv %016llx\n", (unsigned long long)h);
    printf("samples");
    for (int i = 0; i < 11; i++) printf(" %08x", *(const uint32_t *)&o[i]);
    printf("\n");
    ggml_free(ctx);
    return 0;
}
