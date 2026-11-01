// llama.cpp's own MXFP4 arithmetic on fixed inputs: the figures the vendored
// candle's `BlockMxfp4` tests assert (`k_quants.rs`, `mxfp4_tests`).
//
// Build against a llama.cpp checkout's CPU build (no cargo; not an [[example]]):
//
//   R=~/llama.cpp-ref
//   gcc -O2 examples/mxfp4_reference.c -I$R/ggml/include -L$R/build/bin \
//     -lggml-base -lggml-cpu -lm -Wl,-rpath,$R/build/bin -o /tmp/mxfp4_reference
//   /tmp/mxfp4_reference
//
// The inputs come from a xorshift32 the Rust test repeats step for step. It
// prints an FNV-1a hash of every dequantized value's bits, the bits of the dot
// product (`ggml_get_type_traits_cpu(GGML_TYPE_MXFP4)->vec_dot`, AVX2 on an
// AVX2 build) over 8 blocks and over 9 (an odd last block takes the scalar
// rule), and a hash of `quantize_row_mxfp4_ref`'s bytes.
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "ggml-cpu.h"
#include "ggml.h"

static uint32_t s = 0x12345678u;
static uint32_t next(void) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}

static uint64_t fnv1a(const uint8_t *p, size_t n) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < n; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

#define NB 9
#define QK 32

int main(void) {
    ggml_cpu_init();
    // block_mxfp4: { uint8_t e; uint8_t qs[16]; }, 17 bytes.
    uint8_t x[NB * 17];
    for (int b = 0; b < NB; b++) {
        uint8_t e = b == 0 ? 0 : b == 1 ? 1 : (uint8_t)(110 + next() % 30);
        x[b * 17] = e;
        for (int j = 0; j < 16; j++) x[b * 17 + 1 + j] = (uint8_t)(next() & 0xFF);
    }
    // block_q8_0: { ggml_half d; int8_t qs[32]; }, 34 bytes, d little-endian.
    uint8_t y[NB * 34];
    for (int b = 0; b < NB; b++) {
        uint16_t d = (uint16_t)(0x2000 + next() % 0x2000);
        memcpy(&y[b * 34], &d, 2);
        for (int j = 0; j < 32; j++) y[b * 34 + 2 + j] = (uint8_t)(next() & 0xFF);
    }

    const struct ggml_type_traits *tt = ggml_get_type_traits(GGML_TYPE_MXFP4);
    const struct ggml_type_traits_cpu *ct = ggml_get_type_traits_cpu(GGML_TYPE_MXFP4);

    float deq[NB * QK];
    tt->to_float(x, deq, NB * QK);
    printf("dequant_fnv %016llx\n", (unsigned long long)fnv1a((const uint8_t *)deq, sizeof deq));
    printf("dequant_samples");
    for (int i = 0; i < 4; i++) printf(" %08x", *(uint32_t *)&deq[i * 37 % (NB * QK)]);
    printf("\n");

    float dot8 = 0, dot9 = 0;
    ct->vec_dot(8 * QK, &dot8, 0, x, 0, y, 0, 1);
    ct->vec_dot(9 * QK, &dot9, 0, x, 0, y, 0, 1);
    printf("dot8 %08x %.9g\n", *(uint32_t *)&dot8, dot8);
    printf("dot9 %08x %.9g\n", *(uint32_t *)&dot9, dot9);

    // Quantize: four blocks of spread values, one of zeros, one small.
    float src[6 * QK];
    for (int i = 0; i < 4 * QK; i++) src[i] = (float)((int)(next() % 20001) - 10000) / 997.0f;
    for (int i = 4 * QK; i < 5 * QK; i++) src[i] = 0.0f;
    for (int i = 5 * QK; i < 6 * QK; i++) src[i] = (float)((int)(next() % 2001) - 1000) / 65536.0f;
    uint8_t q[6 * 17];
    tt->from_float_ref(src, q, 6 * QK);
    printf("quant_fnv %016llx\n", (unsigned long long)fnv1a(q, sizeof q));
    printf("quant_first");
    for (int i = 0; i < 17; i++) printf(" %02x", q[i]);
    printf("\n");
    return 0;
}
