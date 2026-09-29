// Single-position (decode) attention on the card, straight over the f32 KV cache.
//
// WHY THIS FILE EXISTS
// --------------------
// A decoded token's attention on a card was two cuBLAS matmuls around a
// softmax: ~0.1 ms per layer at 512 positions and 0.32 ms at 8K on an RTX 3070
// (llama-3.2-3b, `layers::gqa_decode_keeps_flash`'s table), i.e. ~3 ms of a
// 28-layer token for a few MB of cache — launch and GEMM overhead, not work.
// candle-flash-attn is no answer for ONE query row: it ships no split-KV
// kernel, so the grid is `n_head` blocks and the card idles (2.5-20x slower
// than the matmuls). On a card-bound 7B this is the biggest cost outside the
// weight matmuls, and llama.cpp — which decodes the same file ~20% faster on
// the same card — does it in one split-KV kernel.
//
// HOW
// ---
// Flash-decoding (Dao, Haziza, Massa, Sizov, "Flash-Decoding for long-context
// inference", 2023; llama.cpp's `fattn-vec` + `flash_attn_combine_results`):
// split the cached positions into fixed chunks, compute each chunk's softmax
// partials in parallel, merge them with the log-sum-exp rescale.
//   decode_attn_f32          grid (batch * n_kv_head, n_chunks), DA_THREADS
//   decode_attn_combine_f32  grid (batch * n_head), only when n_chunks > 1
// One block holds ONE KV head and EVERY query head of its group, so each K and
// V row of its chunk is read once for the group — the rule the CPU kernel
// (`inference::decode_attn`) learned the hard way (#119: once per query head
// was 7x the traffic on Qwen2.5-7B).
//
// The chunk is FIXED (`DA_CHUNK`, mirrored by `decode_attn::CUDA_CHUNK`), so the
// summation order — and the result — depends on the cache length only, never on
// the card or the scheduling. It is not bit-identical to the matmul path it
// replaces (a different summation order): replies are judged against
// llama.cpp, as for the tensor-core prompt pass.
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>
#include <math.h>

#define DA_CHUNK 64
#define DA_MAX_REP 16
#define DA_MAX_D 256
#define DA_THREADS 128
#define DA_WARPS (DA_THREADS / 32)

static __device__ __forceinline__ float da_warp_sum(float x) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, m, 32);
    }
    return x;
}

static __device__ __forceinline__ float da_warp_max(float x) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, m, 32));
    }
    return x;
}

// q:  [b, n_head, d] contiguous, head = kv_head * n_rep + r (repeat_kv's order)
// k,v: the cache VIEW — [b, n_kv_head, s_len, d], rows dense (stride d), batch
//      and head strides as given (a reserved buffer narrowed to s_len)
// out:     [b, n_head, d] — written here only when there is ONE chunk
// partial: [b * n_kv_head, n_chunks, n_rep, d + 2] = (max, Σexp, Σexp·v) per
//          chunk and query head — written only when there are several
// softcap: Gemma-2's tanh cap on the scaled score, 0 = none.
// Requires d % 32 == 0, d <= DA_MAX_D, n_rep <= DA_MAX_REP (checked by the caller).
extern "C" __global__ void decode_attn_f32(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    float *__restrict__ out,
    float *__restrict__ partial,
    const int n_kv_head,
    const int n_rep,
    const int d,
    const int s_len,
    const long long k_sb,
    const long long k_sh,
    const long long v_sb,
    const long long v_sh,
    const float scale,
    const float softcap
) {
    __shared__ float q_sh[DA_MAX_REP * DA_MAX_D];
    __shared__ float sc[DA_MAX_REP * DA_CHUNK];
    __shared__ float red_m[DA_MAX_REP];
    __shared__ float red_l[DA_MAX_REP];

    const int g = blockIdx.x;
    const int bi = g / n_kv_head;
    const int h = g % n_kv_head;
    const int c = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int s0 = c * DA_CHUNK;
    const int n = min(DA_CHUNK, s_len - s0);
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int per_lane = d >> 5;

    // The group's query heads, into shared memory once.
    const float *qg = q + ((long long)g * n_rep) * d;
    for (int i = tid; i < n_rep * d; i += DA_THREADS) {
        q_sh[i] = qg[i];
    }
    __syncthreads();

    const float *kb = k + bi * k_sb + h * k_sh + (long long)s0 * d;
    const float *vb = v + bi * v_sb + h * v_sh + (long long)s0 * d;

    // Scores: one warp per position; its K row is read once, for every head
    // of the group.
    for (int p = warp; p < n; p += DA_WARPS) {
        float kr[DA_MAX_D / 32];
#pragma unroll
        for (int i = 0; i < DA_MAX_D / 32; i++) {
            kr[i] = i < per_lane ? kb[(long long)p * d + lane + 32 * i] : 0.0f;
        }
        for (int r = 0; r < n_rep; r++) {
            float acc = 0.0f;
#pragma unroll
            for (int i = 0; i < DA_MAX_D / 32; i++) {
                if (i < per_lane) {
                    acc += q_sh[r * d + lane + 32 * i] * kr[i];
                }
            }
            acc = da_warp_sum(acc);
            if (lane == 0) {
                float x = acc * scale;
                if (softcap > 0.0f) {
                    x = softcap * tanhf(x / softcap);
                }
                sc[r * DA_CHUNK + p] = x;
            }
        }
    }
    __syncthreads();

    // Max-shifted exp per query head, and its sum over the chunk.
    for (int r = warp; r < n_rep; r += DA_WARPS) {
        float m = -INFINITY;
        for (int p = lane; p < n; p += 32) {
            m = fmaxf(m, sc[r * DA_CHUNK + p]);
        }
        m = da_warp_max(m);
        float l = 0.0f;
        for (int p = lane; p < n; p += 32) {
            const float e = expf(sc[r * DA_CHUNK + p] - m);
            sc[r * DA_CHUNK + p] = e;
            l += e;
        }
        l = da_warp_sum(l);
        if (lane == 0) {
            red_m[r] = m;
            red_l[r] = l;
        }
    }
    __syncthreads();

    // Σ exp·V: each thread owns output dims; each V row is read once for the
    // whole group (consecutive threads read consecutive floats of a row).
    for (int j = tid; j < d; j += DA_THREADS) {
        float acc[DA_MAX_REP];
#pragma unroll
        for (int r = 0; r < DA_MAX_REP; r++) {
            acc[r] = 0.0f;
        }
        for (int p = 0; p < n; p++) {
            const float vv = vb[(long long)p * d + j];
#pragma unroll
            for (int r = 0; r < DA_MAX_REP; r++) {
                if (r < n_rep) {
                    acc[r] += sc[r * DA_CHUNK + p] * vv;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < DA_MAX_REP; r++) {
            if (r < n_rep) {
                if (n_chunks == 1) {
                    out[((long long)g * n_rep + r) * d + j] = acc[r] / red_l[r];
                } else {
                    float *part = partial + (((long long)g * n_chunks + c) * n_rep + r) * (d + 2);
                    part[2 + j] = acc[r];
                    if (j == 0) {
                        part[0] = red_m[r];
                        part[1] = red_l[r];
                    }
                }
            }
        }
    }
}

// Merge the chunks: out = Σ_c e^(m_c - M) o_c / Σ_c e^(m_c - M) l_c — the CPU
// kernel's pass 2, in the same order.
extern "C" __global__ void decode_attn_combine_f32(
    const float *__restrict__ partial,
    float *__restrict__ out,
    const int n_kv_head,
    const int n_rep,
    const int d,
    const int n_chunks
) {
    const int bh = blockIdx.x;
    const int n_head = n_kv_head * n_rep;
    const int bi = bh / n_head;
    const int hh = bh % n_head;
    const int g = bi * n_kv_head + hh / n_rep;
    const int r = hh % n_rep;
    const int row = d + 2;

    float big_m = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        big_m = fmaxf(big_m, partial[((long long)(g * n_chunks + c) * n_rep + r) * row]);
    }
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float num = 0.0f;
        float den = 0.0f;
        for (int c = 0; c < n_chunks; c++) {
            const float *part = partial + ((long long)(g * n_chunks + c) * n_rep + r) * row;
            const float m = part[0];
            if (m == -INFINITY) {
                continue;
            }
            const float w = expf(m - big_m);
            den += w * part[1];
            num += w * part[2 + j];
        }
        out[(long long)bh * d + j] = den > 0.0f ? num / den : 0.0f;
    }
}
