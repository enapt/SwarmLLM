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
// and token): a warp owns state rows (value dimensions `col`; llama.cpp one, here
// four by default, `delta_rule_body`), each lane `D / 32` of each row's elements
// in registers for the whole pass; per token
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
// and the state bit for bit the step's at that row. The rest of a DeltaNet
// step is here too: its convolution + SiLU, its gates and q and k's L2 norm,
// one launch each (the end of this file). The
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
// gates [2, b, seq, nv]    contiguous: g, the log-decay (≤ 0), then beta,
//                          already through its sigmoid — two blocks, as
//                          `ssm_gates_*` write them.
// state [b, nv, D, D]      contiguous, row `col` = value dimension.
// state_out                the new state, same layout — OTHER buffers than
//                          `state` (`SsmState`'s second pair: the state a step
//                          reads survives it, and lives at a fixed address for
//                          a captured step). Never the same buffer: both are
//                          `__restrict__`.
// dst   [b, seq, nv, D]    outputs.
//
// Grid (nv, b, D / (4·C)), block (32, 4): four warps of C rows each a block —
// C = 1 the plain names, C = 4 the `_c4` ones (`delta_rule_body`).
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, 32);
    }
    return x;
}

// A row of a tree of guesses with no parent among the step's rows (#437).
constexpr unsigned int NO_PARENT = 0xffffffffu;

// One row's update of a warp's C state rows `s` (lane `lane` holds elements
// r·32 + lane of each): decay, the delta rule's correction, the write. Leaves
// in `o` the lane's share of each row's output, before the warp's sum and the
// scale. The C rows read the token's q, k and gates once; each row's arithmetic
// is the one-row update's, in its order, so its bits are too.
template <int D, int C>
__device__ __forceinline__ void delta_rule_update(
    float (&s)[C][D / 32],
    float (&o)[C],
    const float *__restrict__ qk,
    const float *__restrict__ v,
    const float *__restrict__ g,
    const float *__restrict__ beta_in,
    const int gs,
    const int nk,
    const int nv,
    const int cap,
    const long long sv_b,
    const long long sv_t,
    const long long sv_h,
    const int bi,
    const int h,
    const int kh,
    const int col0,
    const int lane,
    const int t,
    const unsigned int *__restrict__ slots
) {
    constexpr int R = D / 32;
    const unsigned int row = slots != nullptr ? slots[t] : (unsigned int)t;
    const size_t bt = (size_t)bi * cap + row;
    const float *q_t = qk + (bt * 2 * nk + kh) * D;
    const float *k_t = q_t + (size_t)nk * D;
    const size_t gi = (bt * nv + h) * gs;
    const float decay = expf(g[gi]);
    const float beta = beta_in[gi];
    const float *v_t = v + bi * sv_b + (long long)row * sv_t + h * sv_h + col0;
    float v_col[C];
#pragma unroll
    for (int c = 0; c < C; c++) {
        v_col[c] = v_t[c];
    }

    float kr[R], qr[R];
#pragma unroll
    for (int r = 0; r < R; r++) {
        kr[r] = k_t[r * 32 + lane];
        qr[r] = q_t[r * 32 + lane];
    }

    float kv[C];
#pragma unroll
    for (int c = 0; c < C; c++) {
        kv[c] = 0.0f;
#pragma unroll
        for (int r = 0; r < R; r++) {
            kv[c] += s[c][r] * kr[r];
        }
    }
#pragma unroll
    for (int c = 0; c < C; c++) {
        kv[c] = warp_sum(kv[c]);
    }

#pragma unroll
    for (int c = 0; c < C; c++) {
        const float delta = (v_col[c] - decay * kv[c]) * beta;
        o[c] = 0.0f;
#pragma unroll
        for (int r = 0; r < R; r++) {
            s[c][r] = decay * s[c][r] + kr[r] * delta;
            o[c] += s[c][r] * qr[r];
        }
    }
}

// g and beta of (token bt, head h) are at `g[(bt·nv + h)·gs]` and
// `beta[(bt·nv + h)·gs]`: a step passes its gates' two blocks, a replay the two
// tensors it kept (gs = 1 both). OUT = false writes no outputs: a
// replay wants only the state, and runs every update as the step did, so the
// state is bit for bit the one the step passed at that row.
//
// `parents` (a tree of guesses, #437; `nullptr` for a step's rows in order):
// row t follows row parents[t] (`NO_PARENT`: the state before the step). A row
// that follows the row just read goes on from its state, as a chain's does; any
// other starts again from the state read and walks down its own line, root
// first — the same updates in the same order as the line read alone, so its
// state and output are that line's bits (Twig, arXiv 2608.01651, solves the
// same tree in closed form; at ≤ 32 rows the walk is cheaper than its
// matrices). Ordered best child first, a tree walks again once a branch. The
// state written is the last row's.
//
// `walks` (a tree that FLOWS through the layer, `layers::qwen35::FlowRows`;
// `nullptr` otherwise): the rows each such row walks, root first, named by the
// host — `walks[t]..walks[t + 1]` index `walks[seq + 1 + i]` — where a flowing
// forward's lines, read from the walk's anchor, may be deeper than `line`
// holds.
//
// `slots` (a flowing forward reading its rows where the flow's store keeps
// them, `FlowRows::put`; `nullptr` otherwise): row t's inputs are row
// `slots[t]` of sources `cap` rows a sequence — the gather each kind made
// first, read in place (vLLM's PagedAttention reads its cache through a block
// table the same way). The same values in the same order: the same bits.
// Outputs and walks still name rows 0..seq.
//
// A warp owns C consecutive state rows (`col0` .. `col0 + C`): C = 4 halves
// what a token's inputs cost a row and lets every warp of a layer be on the
// card at once (`delta_rule_*_c4`, grid z D / 16) — at one row a warp, the
// 0.8B's 16 heads × 128 rows are 2,048 warps where an RTX 3070 Laptop holds
// 1,920 of these, so the last blocks ran as a second wave behind the first
// (2026-11-24). The state read is kept in registers: a row that starts again
// from it reads no memory to do so.
template <int D, int C, bool OUT>
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
    const int cap,
    const long long sv_b,
    const long long sv_t,
    const long long sv_h,
    const float scale,
    const unsigned int *__restrict__ parents,
    const unsigned int *__restrict__ walks,
    const unsigned int *__restrict__ slots
) {
    constexpr int R = D / 32;
    const int h = blockIdx.x;
    const int bi = blockIdx.y;
    const int lane = threadIdx.x;
    const int col0 = (blockIdx.z * blockDim.y + threadIdx.y) * C;
    if (col0 >= D) {
        return;
    }
    const int kh = h % nk;

    float read[C][R];
    float s[C][R];
    float o[C];
    const float *s_in = state + (((size_t)bi * nv + h) * D + col0) * D;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            read[c][r] = s_in[c * D + r * 32 + lane];
            s[c][r] = read[c][r];
        }
    }

    for (int t = 0; t < seq; t++) {
        if (parents != nullptr && parents[t] != (unsigned int)(t - 1)) {
#pragma unroll
            for (int c = 0; c < C; c++) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    s[c][r] = read[c][r];
                }
            }
            if (walks != nullptr) {
                const unsigned int *walk = walks + seq + 1;
                for (unsigned int i = walks[t]; i < walks[t + 1]; i++) {
                    delta_rule_update<D, C>(s, o, qk, v, g, beta_in, gs, nk, nv, cap, sv_b,
                                            sv_t, sv_h, bi, h, kh, col0, lane, (int)walk[i],
                                            slots);
                }
            } else {
                int line[32];
                int n = 0;
                for (unsigned int a = parents[t]; a < (unsigned int)t && n < 32;
                     a = parents[a]) {
                    line[n++] = (int)a;
                }
                for (int i = n - 1; i >= 0; i--) {
                    delta_rule_update<D, C>(s, o, qk, v, g, beta_in, gs, nk, nv, cap, sv_b,
                                            sv_t, sv_h, bi, h, kh, col0, lane, line[i], slots);
                }
            }
        }
        delta_rule_update<D, C>(s, o, qk, v, g, beta_in, gs, nk, nv, cap, sv_b, sv_t,
                                sv_h, bi, h, kh, col0, lane, t, slots);
        if constexpr (OUT) {
#pragma unroll
            for (int c = 0; c < C; c++) {
                o[c] = warp_sum(o[c]);
            }
            if (lane == 0) {
                const size_t bt = (size_t)bi * seq + t;
#pragma unroll
                for (int c = 0; c < C; c++) {
                    dst[(bt * nv + h) * D + col0 + c] = o[c] * scale;
                }
            }
        }
    }

    float *s_out = state_out + (((size_t)bi * nv + h) * D + col0) * D;
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            s_out[c * D + r * 32 + lane] = s[c][r];
        }
    }
}

#define DELTA_RULE_ENTRY(D, C, SUF)                                                        \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_f32_d##D##SUF(         \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ gates, const float *__restrict__ state,          \
        float *__restrict__ state_out, float *__restrict__ dst, const int nk,      \
        const int nv, const int seq,                                               \
        const long long sv_b, const long long sv_t, const long long sv_h,          \
        const float scale) {                                                       \
        const float *beta = gates + (size_t)gridDim.y * seq * nv;                  \
        delta_rule_body<D, C, true>(qk, v, gates, beta, 1, state, state_out, dst, nk, \
                                 nv, seq, seq, sv_b, sv_t, sv_h, scale, nullptr,   \
                                 nullptr, nullptr);                                \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_tree_f32_d##D##SUF(    \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ gates, const float *__restrict__ state,          \
        float *__restrict__ state_out, float *__restrict__ dst, const int nk,      \
        const int nv, const int seq,                                               \
        const long long sv_b, const long long sv_t, const long long sv_h,          \
        const float scale, const unsigned int *__restrict__ parents) {             \
        const float *beta = gates + (size_t)gridDim.y * seq * nv;                  \
        delta_rule_body<D, C, true>(qk, v, gates, beta, 1, state, state_out, dst, nk, \
                                 nv, seq, seq, sv_b, sv_t, sv_h, scale, parents,   \
                                 nullptr, nullptr);                                \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_flow_f32_d##D##SUF(    \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ gates, const float *__restrict__ state,          \
        float *__restrict__ state_out, float *__restrict__ dst, const int nk,      \
        const int nv, const int seq,                                               \
        const long long sv_b, const long long sv_t, const long long sv_h,          \
        const float scale, const unsigned int *__restrict__ parents,               \
        const unsigned int *__restrict__ walks) {                                  \
        const float *beta = gates + (size_t)gridDim.y * seq * nv;                  \
        delta_rule_body<D, C, true>(qk, v, gates, beta, 1, state, state_out, dst, nk, \
                                 nv, seq, seq, sv_b, sv_t, sv_h, scale, parents,   \
                                 walks, nullptr);                                  \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_flow_stored_f32_d##D##SUF( \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ g, const float *__restrict__ beta,               \
        const float *__restrict__ state, float *__restrict__ state_out,            \
        float *__restrict__ dst, const int nk, const int nv, const int seq,        \
        const int cap, const long long sv_b, const long long sv_t,                 \
        const long long sv_h, const float scale,                                   \
        const unsigned int *__restrict__ parents,                                  \
        const unsigned int *__restrict__ walks,                                    \
        const unsigned int *__restrict__ slots) {                                  \
        delta_rule_body<D, C, true>(qk, v, g, beta, 1, state, state_out, dst, nk, nv, \
                                 seq, cap, sv_b, sv_t, sv_h, scale, parents,       \
                                 walks, slots);                                    \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) delta_rule_replay_f32_d##D##SUF(  \
        const float *__restrict__ qk, const float *__restrict__ v,                 \
        const float *__restrict__ g, const float *__restrict__ beta,               \
        const float *__restrict__ state, float *__restrict__ state_out,            \
        const int nk, const int nv, const int rows,                                \
        const long long sv_t, const long long sv_h) {                              \
        delta_rule_body<D, C, false>(qk, v, g, beta, 1, state, state_out, nullptr,    \
                                  nk, nv, rows, rows, 0, sv_t, sv_h, 0.0f, nullptr, \
                                  nullptr, nullptr);                               \
    }

DELTA_RULE_ENTRY(64, 1, )
DELTA_RULE_ENTRY(128, 1, )
DELTA_RULE_ENTRY(64, 4, _c4)
DELTA_RULE_ENTRY(128, 4, _c4)

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

// The same over a TREE of guesses (#437, `parents` as the delta rule reads
// them): row t's taps read up its own line — itself, its parent, its parent's
// parent — then the window before the step, last input first
// (`DeltaNetWeights::causal_conv_along`), each product and sum rounded as the
// chain's are, so a row whose line is the rows before it gets the chain's bits
// and every row the composition's. The window written is the chain's, the last
// K − 1 inputs in row order: a branching tree's state is taken back before it
// is read (`SsmState::mark_branched`).
template <int K>
__device__ void ssm_conv_silu_tree_body(
    const float *__restrict__ x, const float *__restrict__ prev,
    const float *__restrict__ w, float *__restrict__ out,
    float *__restrict__ state_out, const int channels, const int split,
    const int seq, const unsigned int *__restrict__ parents,
    const unsigned int *__restrict__ slots) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    constexpr int PAD = K - 1;
    float wc[K];
#pragma unroll
    for (int j = 0; j < K; j++) {
        wc[j] = w[(size_t)c * K + j];
    }
    float *v_out = out + (size_t)seq * split;
    for (int t = 0; t < seq; t++) {
        float win[K];
        unsigned int at = (unsigned int)t;
        int above = 0;
#pragma unroll
        for (int back = 0; back < K; back++) {
            if (at != NO_PARENT) {
                win[PAD - back] = x[(size_t)(slots != nullptr ? slots[at] : at) * channels + c];
                const unsigned int up = parents[at];
                at = up < at ? up : NO_PARENT;
            } else {
                above++;
                win[PAD - back] = prev[(size_t)c * PAD + (PAD - above)];
            }
        }
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
    }
#pragma unroll
    for (int j = 0; j < PAD; j++) {
        const int idx = seq - PAD + j;
        state_out[(size_t)c * PAD + j] =
            idx >= 0 ? x[(size_t)(slots != nullptr ? slots[idx] : (unsigned int)idx) * channels + c]
                     : prev[(size_t)c * PAD + PAD + idx];
    }
}

extern "C" __global__ void ssm_conv_silu_tree_f32_k4(
    const float *__restrict__ x, const float *__restrict__ prev,
    const float *__restrict__ w, float *__restrict__ out,
    float *__restrict__ state_out, const int channels, const int split,
    const int seq, const unsigned int *__restrict__ parents) {
    ssm_conv_silu_tree_body<4>(x, prev, w, out, state_out, channels, split, seq, parents,
                               nullptr);
}

// A flowing forward's convolution reading its rows where the flow's store keeps
// them: row t's input is row `slots[t]` of `x` (`delta_rule_body`'s `slots`).
extern "C" __global__ void ssm_conv_silu_tree_stored_f32_k4(
    const float *__restrict__ x, const float *__restrict__ prev,
    const float *__restrict__ w, float *__restrict__ out,
    float *__restrict__ state_out, const int channels, const int split,
    const int seq, const unsigned int *__restrict__ parents,
    const unsigned int *__restrict__ slots) {
    ssm_conv_silu_tree_body<4>(x, prev, w, out, state_out, channels, split, seq, parents,
                               slots);
}

// ── A step's gates: the decay g and the write strength beta ──
//
// g = softplus(alpha + dt_bias) · a and beta = sigmoid(beta_raw), each
// operation rounded as candle's composition rounds it (`qwen35::softplus`,
// `candle_nn::ops::sigmoid`): relu = fmaxf(x, 0), `+ 1.0` an add, sigmoid
// `recipg(1 + expg(−x))` with `recipg` a DOUBLE division (`1.0 / a`) — bit for
// bit, which a double rounding of a division keeps (53 ≥ 2·24 + 2). Written as
// [g | beta], `[2, rows, nv]`, the block layout the delta rule reads (`gates`
// above): the stack the composition needed is gone too.
__device__ __forceinline__ float gate_beta(const float raw) {
    return 1.0 / (1.0f + expf(-raw));
}

__device__ __forceinline__ float gate_g(const float raw, const float dt, const float a) {
    const float s = __fadd_rn(raw, dt);
    const float sp = __fadd_rn(fmaxf(s, 0.0f), logf(__fadd_rn(expf(-fabsf(s)), 1.0f)));
    return __fmul_rn(sp, a);
}

// After the projections ran as matmuls (a quantized `ssm_alpha` / `ssm_beta`):
// the rest in one launch where the composition made eleven. `n` = rows · nv.
extern "C" __global__ void ssm_gates_f32(
    const float *__restrict__ beta_raw, const float *__restrict__ alpha_raw,
    const float *__restrict__ dt_bias, const float *__restrict__ a,
    float *__restrict__ gates, const int nv, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int h = i % nv;
    gates[i] = gate_g(alpha_raw[i], dt_bias[h], a[h]);
    gates[n + i] = gate_beta(beta_raw[i]);
}

// The projections too, for f32 weights (every popular Qwen 3.5 / 3.6 upload
// keeps `ssm_alpha` and `ssm_beta` in f32) and up to 8 rows — a step or a
// check, where cuBLAS ran a dot product per row and `gemmSN` beyond one
// (+0.47 ms a 2-row check of Qwen3.5-4B). llama.cpp keeps such small batches
// off cuBLAS the same way (`ggml-cuda/mmvf.cu`, up to 8 columns).
//
// A block a (head, projection): grid (nv, 2) — beta then alpha — and 128
// threads, each summing its stride of `hidden` for every row with `fmaf`, then
// a fixed reduction. A row's sum is the same whatever `N`, so a check's gates
// are the gates of its rows read one at a time, bit for bit; NOT bit-identical
// to cuBLAS's. x `[N, hidden]`, w_beta / w_alpha `[nv, hidden]`.
//
// `Q8` reads the weights as Q8_0 blocks instead (34 bytes: an f16 scale d,
// then 32 int8 q; a row is hidden / 32 blocks), as a file that quantized them
// keeps them — Qwen3.5-9B's Q4_K_M upload, and the 0.8B and 2B drafters' Q8_0.
// A weight is d · q, an f16 times an int8: 11 + 8 significant bits, exact in an
// f32, so each weight is the dequantized f32 weight exactly and the sum is the
// f32 kernel's over the dequantized matrix, bit for bit. Where the projections
// ran as matmuls, each was a quantize of x to Q8_1 plus an int8 product (two
// launches a projection, five with the gates); here x stays f32, so NOT
// bit-identical to that path either.
//
// Prior art: llama.cpp's old DMMV kernels read quantized weights against f32
// activations, dequantizing on the fly (`dequantize_mul_mat_vec`, since removed
// for mmvq, which wins on large matrices); for these [nv, hidden] weights the
// activation's quantize costs a launch as long as the product, and SGLang folds
// the gating into its delta-rule update the same way (2026-11-23, #228 (1) (b)).
template <bool Q8>
__device__ __forceinline__ float gate_weight(const void *__restrict__ w, const int h,
                                             const int k, const int hidden) {
    if (Q8) {
        const unsigned char *block = (const unsigned char *)w +
                                     ((size_t)h * (hidden / 32) + k / 32) * 34;
        const float d = __half2float(__ushort_as_half(*(const unsigned short *)block));
        return d * (float)(signed char)block[2 + k % 32];
    }
    return ((const float *)w)[(size_t)h * hidden + k];
}

template <int N, bool Q8>
__device__ void ssm_gates_project_body(
    const float *__restrict__ x, const void *__restrict__ w_beta,
    const void *__restrict__ w_alpha, const float *__restrict__ dt_bias,
    const float *__restrict__ a, float *__restrict__ gates, const int hidden,
    const int nv) {
    const int h = blockIdx.x;
    const bool is_alpha = blockIdx.y == 1;
    const void *w = is_alpha ? w_alpha : w_beta;
    float acc[N];
#pragma unroll
    for (int t = 0; t < N; t++) {
        acc[t] = 0.0f;
    }
    for (int k = threadIdx.x; k < hidden; k += 128) {
        const float wk = gate_weight<Q8>(w, h, k, hidden);
#pragma unroll
        for (int t = 0; t < N; t++) {
            acc[t] = fmaf(wk, x[(size_t)t * hidden + k], acc[t]);
        }
    }
    __shared__ float part[N][4];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
#pragma unroll
    for (int t = 0; t < N; t++) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) {
            part[t][warp] = s;
        }
    }
    __syncthreads();
    if (threadIdx.x < N) {
        const int t = threadIdx.x;
        const float raw = __fadd_rn(__fadd_rn(part[t][0], part[t][1]),
                                    __fadd_rn(part[t][2], part[t][3]));
        const int i = t * nv + h;
        if (is_alpha) {
            gates[i] = gate_g(raw, dt_bias[h], a[h]);
        } else {
            gates[N * nv + i] = gate_beta(raw);
        }
    }
}

#define SSM_GATES_PROJECT_ENTRY(N)                                                 \
    extern "C" __global__ void __launch_bounds__(128) ssm_gates_project_f32_r##N(  \
        const float *__restrict__ x, const float *__restrict__ w_beta,             \
        const float *__restrict__ w_alpha, const float *__restrict__ dt_bias,      \
        const float *__restrict__ a, float *__restrict__ gates, const int hidden,  \
        const int nv) {                                                            \
        ssm_gates_project_body<N, false>(x, w_beta, w_alpha, dt_bias, a, gates,    \
                                         hidden, nv);                              \
    }                                                                              \
    extern "C" __global__ void __launch_bounds__(128) ssm_gates_project_q8_0_r##N( \
        const float *__restrict__ x, const unsigned char *__restrict__ w_beta,     \
        const unsigned char *__restrict__ w_alpha,                                 \
        const float *__restrict__ dt_bias, const float *__restrict__ a,            \
        float *__restrict__ gates, const int hidden, const int nv) {               \
        ssm_gates_project_body<N, true>(x, w_beta, w_alpha, dt_bias, a, gates,     \
                                        hidden, nv);                               \
    }

SSM_GATES_PROJECT_ENTRY(1)
SSM_GATES_PROJECT_ENTRY(2)
SSM_GATES_PROJECT_ENTRY(3)
SSM_GATES_PROJECT_ENTRY(4)
SSM_GATES_PROJECT_ENTRY(5)
SSM_GATES_PROJECT_ENTRY(6)
SSM_GATES_PROJECT_ENTRY(7)
SSM_GATES_PROJECT_ENTRY(8)

// ── q and k's L2 norm: x / sqrt(Σx² + eps) a row (`qwen35::l2_norm`) ──
//
// One warp a row of D, in one launch where the composition made five (square,
// sum, add, root, divide). The sum runs in another order than candle's, so
// NOT bit-identical; the add, the root and the division are its. A row reads
// only itself, so a check's rows are its rows read one at a time. llama.cpp
// runs its L2 norm as one launch a row too (`ggml-cuda/norm.cu`).
// Block (32, 4): four rows a block.
template <int D>
__device__ void l2_norm_rows_body(
    const float *__restrict__ x, float *__restrict__ out, const int rows,
    const float eps) {
    constexpr int R = D / 32;
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= rows) {
        return;
    }
    const int lane = threadIdx.x;
    const float *xr = x + (size_t)row * D;
    float v[R];
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < R; r++) {
        v[r] = xr[r * 32 + lane];
        s = fmaf(v[r], v[r], s);
    }
    s = warp_sum(s);
    const float d = sqrtf(__fadd_rn(s, eps));
    float *o = out + (size_t)row * D;
#pragma unroll
    for (int r = 0; r < R; r++) {
        o[r * 32 + lane] = v[r] / d;
    }
}

extern "C" __global__ void __launch_bounds__(128) l2_norm_rows_f32_d64(
    const float *__restrict__ x, float *__restrict__ out, const int rows,
    const float eps) {
    l2_norm_rows_body<64>(x, out, rows, eps);
}

extern "C" __global__ void __launch_bounds__(128) l2_norm_rows_f32_d128(
    const float *__restrict__ x, float *__restrict__ out, const int rows,
    const float eps) {
    l2_norm_rows_body<128>(x, out, rows, eps);
}

// A flowing forward's new rows written into its store (`qwen35::FlowRows::put`):
// row r of `src` [rows, width] to row `slots[r]` of `store` [capacity, width] —
// the slots the walk left are written again (`FlowRows::new_slots`), so the
// slots are not one run, and ONE launch a kind covers every pattern of them:
// a captured level is one topology however its rows land. A copy: the same
// bits as `slice_set`. One sequence. Grid (rows), block 256; a slot past the
// store's rows is checked on the host (`FlowRows::rows_the_lines_read`) and
// skipped here.
extern "C" __global__ void flow_rows_put_f32(
    const float *__restrict__ src, float *__restrict__ store,
    const unsigned int *__restrict__ slots, const int width, const int capacity) {
    const unsigned int slot = slots[blockIdx.x];
    if (slot >= (unsigned int)capacity) {
        return;
    }
    const float *from = src + (size_t)blockIdx.x * width;
    float *to = store + (size_t)slot * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        to[i] = from[i];
    }
}
