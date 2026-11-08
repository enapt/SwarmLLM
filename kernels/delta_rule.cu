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
// — the decay folded into the update as llama.cpp's card op folds it.
//
// A REPLAY (`SsmState::rewind_to` into a check, `layers::qwen35::replay_rows`)
// runs the same body over the first rows a check kept, outputs off
// (`delta_rule_replay_f32_d*`), and `conv_window_replay_f32` rebuilds the
// convolution's window: two launches a layer where the composition made seven,
// and the state bit for bit the step's at that row. The
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
// state_out                the new state, same layout — OTHER buffers than
//                          `state` (`SsmState`'s second pair: the state a step
//                          reads survives it, and lives at a fixed address for
//                          a captured step). Never the same buffer: both are
//                          `__restrict__`.
// dst   [b, seq, nv, D]    outputs.
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

// g and beta of (token bt, head h) are at `g[(bt·nv + h)·gs]` and
// `beta[(bt·nv + h)·gs]`: a step passes its stacked `gb` (beta = gb + 1, gs = 2),
// a replay the two rows it kept (gs = 1). OUT = false writes no outputs: a
// replay wants only the state, and runs every update as the step did, so the
// state is bit for bit the one the step passed at that row.
template <int D, bool OUT>
__device__ void delta_rule_body(
    const float *__restrict__ qk,
    const float *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta_in,
    const int gs,
    const float *__restrict__ state,
    float *__restrict__ state_out,
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
        const size_t gi = (bt * nv + h) * gs;
        const float decay = expf(g[gi]);
        const float beta = beta_in[gi];
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
        if constexpr (OUT) {
            o = warp_sum(o);
            if (lane == 0) {
                dst[(bt * nv + h) * D + col] = o * scale;
            }
        }
    }

    float *s_out = state_out + (((size_t)bi * nv + h) * D + col) * D;
#pragma unroll
    for (int r = 0; r < R; r++) {
        s_out[r * 32 + lane] = s[r];
    }
}

#define DELTA_RULE_ENTRY(D)                                                        \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_f32_d##D(         \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ gb, const float *__restrict__ state,             \
        float *__restrict__ state_out, float *__restrict__ dst, const int nk,      \
        const int nv, const int seq,                                               \
        const long long sv_b, const long long sv_t, const long long sv_h,          \
        const float scale) {                                                       \
        delta_rule_body<D, true>(qk, v, gb, gb + 1, 2, state, state_out, dst, nk,  \
                                 nv, seq, sv_b, sv_t, sv_h, scale);                \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_replay_f32_d##D(  \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ g, const float *__restrict__ beta,               \
        const float *__restrict__ state, float *__restrict__ state_out,            \
        const int nk, const int nv, const int rows,                                \
        const long long sv_t, const long long sv_h) {                              \
        delta_rule_body<D, false>(qk, v, g, beta, 1, state, state_out, nullptr,    \
                                  nk, nv, rows, 0, sv_t, sv_h, 0.0f);              \
    }

DELTA_RULE_ENTRY(64)
DELTA_RULE_ENTRY(128)

// A replay's convolution window (`SsmState::rewind_to` into a check): the last
// `pad` (= kernel − 1) of [the window the check read | its first `rows` inputs]
// — what `cat` + `narrow` composed in five launches, as one copy. One sequence:
// `read` and `out` [C, pad], `mixed` the kept inputs [rows ≥ `rows`, C].
extern "C" __global__ void conv_window_replay_f32(
    const float *__restrict__ read, const float *__restrict__ mixed,
    float *__restrict__ out, const int channels, const int pad, const int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= channels * pad) {
        return;
    }
    const int c = i / pad;
    const int src = rows + i % pad;
    out[i] = src < pad ? read[c * pad + src] : mixed[(size_t)(src - pad) * channels + c];
}

// A step's causal depthwise convolution and its SiLU in ONE launch a layer
// (`DeltaNetWeights::causal_conv` + `silu`: 12 launches at one position, up to
// 14 at several — 24 layers of Qwen3.5-4B made ~290 a token). llama.cpp fuses
// the same pair on a card (`ggml_cuda_op_ssm_conv`, apply_silu).
//
// A thread a channel slides its window over the step: output t is
// Σ_j win[t + j] · w[j] over [the window before | the step's inputs], summed
// in tap order with each product and sum rounded as candle's `bmul` / `badd`
// round them (`__fmul_rn` / `__fadd_rn`: no fused multiply-add), then candle's
// `silu_fwd` (`x / (1 + expf(-x))`) — bit for bit the composition. Channels
// below `split` (q | k) and the rest (v) are written to two contiguous blocks
// of `out`, [seq, split] then [seq, C − split]; the window after the step, the
// last K − 1 inputs, straight into `state_out` [C, K − 1], the step's buffers.
// One sequence: `x` [seq, C], `prev` [C, K − 1], `w` [C, K].
template <int K>
__device__ void ssm_conv_silu_body(
    const float *__restrict__ x, const float *__restrict__ prev,
    const float *__restrict__ w, float *__restrict__ out,
    float *__restrict__ state_out, const int channels, const int split,
    const int seq) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    constexpr int PAD = K - 1;
    float win[K];
    float wc[K];
#pragma unroll
    for (int j = 0; j < K; j++) {
        wc[j] = w[(size_t)c * K + j];
    }
#pragma unroll
    for (int j = 0; j < PAD; j++) {
        win[j] = prev[(size_t)c * PAD + j];
    }
    float *v_out = out + (size_t)seq * split;
    for (int t = 0; t < seq; t++) {
        win[PAD] = x[(size_t)t * channels + c];
        float acc = __fmul_rn(win[0], wc[0]);
#pragma unroll
        for (int j = 1; j < K; j++) {
            acc = __fadd_rn(acc, __fmul_rn(win[j], wc[j]));
        }
        const float y = acc / (1.0f + expf(-acc));
        if (c < split) {
            out[(size_t)t * split + c] = y;
        } else {
            v_out[(size_t)t * (channels - split) + (c - split)] = y;
        }
#pragma unroll
        for (int j = 0; j < PAD; j++) {
            win[j] = win[j + 1];
        }
    }
#pragma unroll
    for (int j = 0; j < PAD; j++) {
        state_out[(size_t)c * PAD + j] = win[j];
    }
}

extern "C" __global__ void ssm_conv_silu_f32_k4(
    const float *__restrict__ x, const float *__restrict__ prev,
    const float *__restrict__ w, float *__restrict__ out,
    float *__restrict__ state_out, const int channels, const int split,
    const int seq) {
    ssm_conv_silu_body<4>(x, prev, w, out, state_out, channels, split, seq);
}
