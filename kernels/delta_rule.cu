// Qwen 3.5's gated delta rule on a card: every token of a layer in ONE launch.
//
// WHY THIS FILE EXISTS
// --------------------
// Three layers in four of a Qwen 3.5 model are Gated DeltaNet layers, whose
// "attention" is a recurrence over a [v_dim, k_dim] state per value head. Composed
// from candle ops it is ~10 launches per token per layer (`layers::qwen35::
// delta_rule`): a 681-token prompt on the 4B made ~160,000 of them, and a decode
// step on a card is bound by how many launches it makes (`docs/invariants/
// inference.md` § "A decode token is bound by GPU submission COUNT").
//
// The kernel is llama.cpp's `gated_delta_net_cuda` (ggml-cuda/gated_delta_net.cu,
// master `4b1a27f`, the scalar-gate case — Qwen 3.5's `g` is one value per head
// and token): a warp owns one state row (one value dimension `col`), each lane
// `D / 32` of its elements in registers for the whole pass; per token
//
//     kv    = Σ_i S[col, i] · k[i]                 (warp sum)
//     delta = (v[col] − exp(g) · kv) · beta
//     S[col, :] = exp(g) · S[col, :] + k · delta
//     out[col]  = scale · Σ_i S[col, i] · q[i]     (warp sum)
//
// — the decay folded into the update as llama.cpp's card op folds it. The
// processor's kernel (`delta_rule_cpu`) decays first; the two agree to float
// rounding, NOT bitwise, and neither is bit-identical to the composition. Replies
// are judged against llama.cpp.
//
// LAYOUT
// ------
// qk    [b, seq, 2·nk, D]  contiguous; heads 0..nk are q, nk..2nk are k, both
//                          L2-normalised. Value head h reads key head h % nk
//                          (llama.cpp tiles the key heads).
// v     element (bi, t, h, j) at bi·sv_b + t·sv_t + h·sv_h + j — a view into the
//                          convolution's output, unit inner stride.
// gb    [b, seq, nv, 2]    contiguous: (g, beta) — g the log-decay (≤ 0), beta
//                          already through its sigmoid.
// state [b, nv, D, D]      contiguous, row `col` = value dimension.
// dst   [b, seq, nv, D] outputs, then [b, nv, D, D] the new state.
//
// Grid (nv, b, D / 4), block (32, 4): four warps, four rows, per block.
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

template <int D>
__device__ void delta_rule_body(
    const float *__restrict__ qk,
    const float *__restrict__ v,
    const float *__restrict__ gb,
    const float *__restrict__ state,
    float *__restrict__ dst,
    const int nk,
    const int nv,
    const int seq,
    const long long sv_b,
    const long long sv_t,
    const long long sv_h,
    const float scale
) {
    constexpr int R = D / 32;
    const int h = blockIdx.x;
    const int bi = blockIdx.y;
    const int lane = threadIdx.x;
    const int col = blockIdx.z * blockDim.y + threadIdx.y;
    if (col >= D) {
        return;
    }
    const int kh = h % nk;
    const size_t n_out = (size_t)gridDim.y * seq * nv * D;

    float s[R];
    const float *s_in = state + (((size_t)bi * nv + h) * D + col) * D;
#pragma unroll
    for (int r = 0; r < R; r++) {
        s[r] = s_in[r * 32 + lane];
    }

    for (int t = 0; t < seq; t++) {
        const size_t bt = (size_t)bi * seq + t;
        const float *q_t = qk + (bt * 2 * nk + kh) * D;
        const float *k_t = q_t + (size_t)nk * D;
        const float *gb_t = gb + (bt * nv + h) * 2;
        const float decay = expf(gb_t[0]);
        const float beta = gb_t[1];
        const float v_col = v[bi * sv_b + t * sv_t + h * sv_h + col];

        float kr[R], qr[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            kr[r] = k_t[r * 32 + lane];
            qr[r] = q_t[r * 32 + lane];
        }

        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < R; r++) {
            kv += s[r] * kr[r];
        }
        kv = warp_sum(kv);

        const float delta = (v_col - decay * kv) * beta;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < R; r++) {
            s[r] = decay * s[r] + kr[r] * delta;
            o += s[r] * qr[r];
        }
        o = warp_sum(o);
        if (lane == 0) {
            dst[(bt * nv + h) * D + col] = o * scale;
        }
    }

    float *s_out = dst + n_out + (((size_t)bi * nv + h) * D + col) * D;
#pragma unroll
    for (int r = 0; r < R; r++) {
        s_out[r * 32 + lane] = s[r];
    }
}

#define DELTA_RULE_ENTRY(D)                                                        \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_f32_d##D(         \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ gb, const float *__restrict__ state,             \
        float *__restrict__ dst, const int nk, const int nv, const int seq,        \
        const long long sv_b, const long long sv_t, const long long sv_h,          \
        const float scale) {                                                       \
        delta_rule_body<D>(qk, v, gb, state, dst, nk, nv, seq, sv_b, sv_t, sv_h,   \
                           scale);                                                 \
    }

DELTA_RULE_ENTRY(64)
DELTA_RULE_ENTRY(128)
