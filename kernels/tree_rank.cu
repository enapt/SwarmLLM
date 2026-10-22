// Rank a drafter's tree level on the card: each row's log-sum-exp and its
// eight likeliest tokens, in two launches and one read of the logits.
//
// WHY THIS FILE EXISTS
// --------------------
// Every level of a drafter's tree ends with its rows ranked: for each row of a
// `[rows, vocab]` logits tensor (vocab 151,936 for a Qwen drafter), the eight
// likeliest tokens and their log-probabilities. Composed from candle ops
// (`split::draft::likeliest_rows_composed`) that is a log-sum-exp (max, sub,
// exp, sum, log), eight rounds of argmax + gather + scatter, and three reads
// back — about thirty launches and three waits on the card a level. candle's
// argmax gives each ROW one block, so eight rows keep eight of a laptop card's
// 40 multiprocessors busy: 134 µs a launch, 21% of all the card's time in a
// drafter's three-level call (nsys, 0.5B Q8_0 on an RTX 3070 Laptop,
// 2026-10-22). Here each row is read once by up to 64 blocks, each thread
// keeping its own top eight, then merged — the shape top-k sampling takes in
// FlashInfer and llama.cpp (a partial top-k per block, merged after).
//
// NOT bit-identical to the composition: the log-sum-exp sums in another order,
// and two equal scores go to the smaller id (candle's argmax breaks a tie its
// own way). The ranking decides only what is GUESSED — every guess is checked by
// the target — never what the reply is. The card test
// (`split::draft::tests::the_card_ranks_a_level_as_the_host_does`) holds it to
// the host's ranking.
//
// LAYOUT
// ------
// `logits` is `[rows, vocab]` f32, contiguous, `vocab < 2^24` (ids go back as
// exact floats; the host checks). Each score is multiplied by `inv_t` (the
// composition's `scaled / t` is candle's `affine(1 / t, 0)`).
//
// Pass 1 (`tree_rank_partials`): grid (chunks, rows), PASS1_THREADS a block.
// Block (c, r) reads chunk c of row r and writes PARTIAL_WIDTH floats at
// `partials[(r * chunks + c) * PARTIAL_WIDTH]`: the chunk's max, its sum of
// exp(x - max), a flag (1 where it held a NaN or +inf), TOPK values likeliest
// first, then their TOPK ids bit-cast.
//
// Pass 2 (`tree_rank_merge`): grid rows, PASS2_THREADS a block, one thread a
// chunk (chunks <= PASS2_THREADS). Writes OUT_WIDTH floats a row at
// `out[r * OUT_WIDTH]`: the row's log-sum-exp, TOPK log-probabilities
// likeliest first, then the TOPK ids as exact floats. A slot past the row's
// finite scores is -inf with id 0. A row with a NaN or +inf, or with nothing
// finite, writes a NaN log-sum-exp: the host ranks those rows itself.
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdint.h>
#include <limits.h>

#define TOPK 8
#define PARTIAL_WIDTH (3 + 2 * TOPK)
#define OUT_WIDTH (1 + 2 * TOPK)
#define PASS1_THREADS 256
#define PASS2_THREADS 64

// `a` ranks before `b`: the higher score, a tie to the smaller id.
__device__ __forceinline__ bool ranks_before(float a, int ia, float b, int ib) {
    return a > b || (a == b && ia < ib);
}

// A thread's likeliest TOPK, best first. Every index is a compile-time
// constant after unrolling, so the lists stay in registers.
struct TopK {
    float v[TOPK];
    int id[TOPK];

    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            v[j] = -CUDART_INF_F;
            id[j] = INT_MAX;
        }
    }

    // Insert (x, i) where it ranks. Never handed -inf: an empty slot is
    // (-inf, INT_MAX), which every finite score ranks before.
    __device__ __forceinline__ void offer(float x, int i) {
        if (!ranks_before(x, i, v[TOPK - 1], id[TOPK - 1])) {
            return;
        }
#pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            if (ranks_before(x, i, v[j], id[j])) {
                const float tv = v[j];
                const int ti = id[j];
                v[j] = x;
                id[j] = i;
                x = tv;
                i = ti;
            }
        }
    }

    __device__ __forceinline__ void drop_head() {
#pragma unroll
        for (int j = 0; j < TOPK - 1; ++j) {
            v[j] = v[j + 1];
            id[j] = id[j + 1];
        }
        v[TOPK - 1] = -CUDART_INF_F;
        id[TOPK - 1] = INT_MAX;
    }
};

// (m, s) += (m2, s2), where s is a sum of exp(x - m). An empty side has
// m = -inf and s = 0.
__device__ __forceinline__ void lse_merge(float &m, float &s, float m2, float s2) {
    if (m2 == -CUDART_INF_F) {
        return;
    }
    if (m == -CUDART_INF_F) {
        m = m2;
        s = s2;
    } else if (m2 > m) {
        s = s * expf(m - m2) + s2;
        m = m2;
    } else {
        s += s2 * expf(m2 - m);
    }
}

// The block's (max, sum, flag), answered in thread 0. `shared` holds at least
// 3 floats a warp.
__device__ __forceinline__ void block_lse(float &m, float &s, int &bad, float *shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float m2 = __shfl_down_sync(0xffffffffu, m, off);
        const float s2 = __shfl_down_sync(0xffffffffu, s, off);
        bad |= __shfl_down_sync(0xffffffffu, bad, off);
        lse_merge(m, s, m2, s2);
    }
    if (lane == 0) {
        shared[3 * warp] = m;
        shared[3 * warp + 1] = s;
        shared[3 * warp + 2] = (float)bad;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < warps; ++w) {
            lse_merge(m, s, shared[3 * w], shared[3 * w + 1]);
            bad |= shared[3 * w + 2] != 0.0f;
        }
    }
    __syncthreads();
}

// Take the block's best TOPK out of every thread's list, best first, into
// thread 0's `best_v` / `best_id`. Each round finds the best head and the one
// thread holding it drops it. `shared` holds at least 2 words a warp + 2.
__device__ __forceinline__ void block_top(TopK &mine, float *best_v, int *best_id, float *shared) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = (blockDim.x + 31) >> 5;
    int *shared_id = reinterpret_cast<int *>(shared);
    for (int round = 0; round < TOPK; ++round) {
        float v = mine.v[0];
        int id = mine.id[0];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float v2 = __shfl_down_sync(0xffffffffu, v, off);
            const int id2 = __shfl_down_sync(0xffffffffu, id, off);
            if (ranks_before(v2, id2, v, id)) {
                v = v2;
                id = id2;
            }
        }
        if (lane == 0) {
            shared[2 * warp] = v;
            shared_id[2 * warp + 1] = id;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < warps; ++w) {
                if (ranks_before(shared[2 * w], shared_id[2 * w + 1], v, id)) {
                    v = shared[2 * w];
                    id = shared_id[2 * w + 1];
                }
            }
            best_v[round] = v;
            best_id[round] = id;
            shared[2 * warps] = v;
            shared_id[2 * warps + 1] = id;
        }
        __syncthreads();
        // Ids are unique within a row (each index has one owner), so only the
        // holder drops. An empty winner means every list is empty.
        const int won = shared_id[2 * warps + 1];
        if (won != INT_MAX && mine.id[0] == won) {
            mine.drop_head();
        }
        __syncthreads();
    }
}

extern "C" __global__ void tree_rank_partials(
    const float *__restrict__ logits,
    float *__restrict__ partials,
    const int vocab,
    const int chunks,
    const float inv_t
) {
    __shared__ float shared[3 * (PASS1_THREADS / 32) + 2];
    const int chunk = blockIdx.x;
    const int row = blockIdx.y;
    const int chunk_len = (vocab + chunks - 1) / chunks;
    const int start = chunk * chunk_len;
    const int end = min(start + chunk_len, vocab);
    const float *scores = logits + (size_t)row * vocab;

    float m = -CUDART_INF_F;
    float s = 0.0f;
    int bad = 0;
    TopK mine;
    mine.clear();
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const float x = scores[i] * inv_t;
        if (isnan(x) || x == CUDART_INF_F) {
            bad = 1;
            continue;
        }
        if (x == -CUDART_INF_F) {
            continue;
        }
        if (x > m) {
            s = s * expf(m - x) + 1.0f;
            m = x;
        } else {
            s += expf(x - m);
        }
        mine.offer(x, i);
    }

    block_lse(m, s, bad, shared);
    float *dst = partials + ((size_t)row * chunks + chunk) * PARTIAL_WIDTH;
    if (threadIdx.x == 0) {
        dst[0] = m;
        dst[1] = s;
        dst[2] = (float)bad;
    }
    float best_v[TOPK];
    int best_id[TOPK];
    block_top(mine, best_v, best_id, shared);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            dst[3 + j] = best_v[j];
            dst[3 + TOPK + j] = __int_as_float(best_id[j]);
        }
    }
}

extern "C" __global__ void tree_rank_merge(
    const float *__restrict__ partials,
    float *__restrict__ out,
    const int chunks
) {
    __shared__ float shared[3 * (PASS2_THREADS / 32) + 2];
    const int row = blockIdx.x;
    float m = -CUDART_INF_F;
    float s = 0.0f;
    int bad = 0;
    TopK mine;
    mine.clear();
    if ((int)threadIdx.x < chunks) {
        const float *src = partials + ((size_t)row * chunks + threadIdx.x) * PARTIAL_WIDTH;
        m = src[0];
        s = src[1];
        bad = src[2] != 0.0f;
#pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            mine.v[j] = src[3 + j];
            mine.id[j] = __float_as_int(src[3 + TOPK + j]);
        }
    }

    block_lse(m, s, bad, shared);
    float best_v[TOPK];
    int best_id[TOPK];
    block_top(mine, best_v, best_id, shared);
    if (threadIdx.x == 0) {
        float *dst = out + (size_t)row * OUT_WIDTH;
        const float lse = (bad || m == -CUDART_INF_F) ? CUDART_NAN_F : m + logf(s);
        dst[0] = lse;
#pragma unroll
        for (int j = 0; j < TOPK; ++j) {
            const bool empty = best_id[j] == INT_MAX;
            dst[1 + j] = empty ? -CUDART_INF_F : best_v[j] - lse;
            dst[1 + TOPK + j] = empty ? 0.0f : (float)best_id[j];
        }
    }
}
