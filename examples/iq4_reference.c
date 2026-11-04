// llama.cpp's own IQ4_NL and IQ4_XS arithmetic on fixed inputs: the figures
// the vendored candle's `BlockIq4Nl` / `BlockIq4Xs` tests assert
// (`src/inference/mod.rs`, `iq4_tests`; #404).
//
// Build against a llama.cpp checkout's CPU build (no cargo; not an [[example]]):
//
//   R=~/llama.cpp-ref
//   gcc -O2 examples/iq4_reference.c -I$R/ggml/include -L$R/build/bin \
//     -lggml-base -lggml-cpu -lm -Wl,-rpath,$R/build/bin -o /tmp/iq4_reference
//   /tmp/iq4_reference
//
// The inputs come from a xorshift32 the Rust test repeats step for step. For
// each type it prints an FNV-1a hash of every dequantized value's bits, the
// bits of the dot product (`ggml_get_type_traits_cpu(..)->vec_dot`, AVX2 on an
// AVX2 build) — IQ4_NL over 8 blocks and over 9 (an odd last block takes the
// scalar rule), IQ4_XS over 1 super-block and over 3 — and a hash of the
// reference quantizer's bytes (`from_float_ref`). The Q8 bytes stay within
// +-127, as a quantizer writes them: llama.cpp's AVX2 sign trick
// (`_mm256_sign_epi8`) reads -128 beside a negative weight wrongly.
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

static int8_t q8(void) { return (int8_t)((int)(next() % 255) - 127); }

#define NB 9
#define QK 32
#define NBX 3
#define QKK 256

static void quantize_and_print(enum ggml_type type, int n, size_t bytes, const char *name) {
    const struct ggml_type_traits *tt = ggml_get_type_traits(type);
    // Spread values, then zeros, then small ones, then a block with one large
    // outlier — the shape that moves a scale's search.
    float src[4 * QKK];
    int quarter = n / 4;
    for (int i = 0; i < quarter; i++) src[i] = (float)((int)(next() % 20001) - 10000) / 997.0f;
    for (int i = quarter; i < 2 * quarter; i++) src[i] = 0.0f;
    for (int i = 2 * quarter; i < 3 * quarter; i++) src[i] = (float)((int)(next() % 2001) - 1000) / 65536.0f;
    for (int i = 3 * quarter; i < n; i++) src[i] = (float)((int)(next() % 2001) - 1000) / 500.0f;
    src[3 * quarter + 5] = 37.5f;
    uint8_t q[4 * 136 * 8];
    tt->from_float_ref(src, q, n);
    printf("%s_quant_fnv %016llx\n", name, (unsigned long long)fnv1a(q, bytes));
    printf("%s_quant_first", name);
    for (int i = 0; i < 18; i++) printf(" %02x", q[i]);
    printf("\n");
}

int main(void) {
    ggml_cpu_init();

    // ---- IQ4_NL: { ggml_half d; uint8_t qs[16]; }, 18 bytes.
    uint8_t x[NB * 18];
    for (int b = 0; b < NB; b++) {
        uint16_t d = (uint16_t)(0x2000 + next() % 0x2000);
        if (b == 2) d |= 0x8000;
        memcpy(&x[b * 18], &d, 2);
        for (int j = 0; j < 16; j++) x[b * 18 + 2 + j] = (uint8_t)(next() & 0xFF);
    }
    // block_q8_0: { ggml_half d; int8_t qs[32]; }, 34 bytes, d little-endian.
    uint8_t y[NB * 34];
    for (int b = 0; b < NB; b++) {
        uint16_t d = (uint16_t)(0x2000 + next() % 0x2000);
        memcpy(&y[b * 34], &d, 2);
        for (int j = 0; j < 32; j++) y[b * 34 + 2 + j] = (uint8_t)q8();
    }
    const struct ggml_type_traits *tt = ggml_get_type_traits(GGML_TYPE_IQ4_NL);
    const struct ggml_type_traits_cpu *ct = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_NL);
    float deq[NB * QK];
    tt->to_float(x, deq, NB * QK);
    printf("nl_dequant_fnv %016llx\n", (unsigned long long)fnv1a((const uint8_t *)deq, sizeof deq));
    float dot8 = 0, dot9 = 0;
    ct->vec_dot(8 * QK, &dot8, 0, x, 0, y, 0, 1);
    ct->vec_dot(9 * QK, &dot9, 0, x, 0, y, 0, 1);
    printf("nl_dot8 %08x %.9g\n", *(uint32_t *)&dot8, dot8);
    printf("nl_dot9 %08x %.9g\n", *(uint32_t *)&dot9, dot9);

    // ---- IQ4_XS: { ggml_half d; uint16_t scales_h; uint8_t scales_l[4];
    // uint8_t qs[128]; }, 136 bytes.
    uint8_t xs[NBX * 136];
    for (int b = 0; b < NBX; b++) {
        uint16_t d = (uint16_t)(0x2000 + next() % 0x2000);
        if (b == 1) d |= 0x8000;
        memcpy(&xs[b * 136], &d, 2);
        uint16_t h = (uint16_t)(next() & 0xFFFF);
        memcpy(&xs[b * 136 + 2], &h, 2);
        for (int j = 0; j < 4; j++) xs[b * 136 + 4 + j] = (uint8_t)(next() & 0xFF);
        for (int j = 0; j < 128; j++) xs[b * 136 + 8 + j] = (uint8_t)(next() & 0xFF);
    }
    // block_q8_K: { float d; int8_t qs[256]; int16_t bsums[16]; }, 292 bytes.
    uint8_t yk[NBX * 292];
    for (int b = 0; b < NBX; b++) {
        float d = (float)(1 + next() % 1000) / 16384.0f;
        memcpy(&yk[b * 292], &d, 4);
        int8_t *qs = (int8_t *)&yk[b * 292 + 4];
        for (int j = 0; j < QKK; j++) qs[j] = q8();
        for (int g = 0; g < 16; g++) {
            int16_t sum = 0;
            for (int j = 0; j < 16; j++) sum += qs[16 * g + j];
            memcpy(&yk[b * 292 + 4 + QKK + 2 * g], &sum, 2);
        }
    }
    tt = ggml_get_type_traits(GGML_TYPE_IQ4_XS);
    ct = ggml_get_type_traits_cpu(GGML_TYPE_IQ4_XS);
    float deqx[NBX * QKK];
    tt->to_float(xs, deqx, NBX * QKK);
    printf("xs_dequant_fnv %016llx\n", (unsigned long long)fnv1a((const uint8_t *)deqx, sizeof deqx));
    float dot1 = 0, dot3 = 0;
    ct->vec_dot(1 * QKK, &dot1, 0, xs, 0, yk, 0, 1);
    ct->vec_dot(3 * QKK, &dot3, 0, xs, 0, yk, 0, 1);
    printf("xs_dot1 %08x %.9g\n", *(uint32_t *)&dot1, dot1);
    printf("xs_dot3 %08x %.9g\n", *(uint32_t *)&dot3, dot3);

    // ---- The reference quantizers: 8 IQ4_NL blocks, 4 IQ4_XS super-blocks.
    quantize_and_print(GGML_TYPE_IQ4_NL, 8 * QK, 8 * 18, "nl");
    quantize_and_print(GGML_TYPE_IQ4_XS, 4 * QKK, 4 * 136, "xs");
    return 0;
}
