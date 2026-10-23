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
// A ROW'S TOP K FOR A COUPLED DRAW (`tree_topk_partials`, `tree_topk_merge`)
// --------------------------------------------------------------------------
// A request that samples draws each guess with the request's shared noise
// (`inference::coupled_noise`), and the sampler draws among the row's `top_k`
// likeliest tokens only (`sampling::sample_among_top_k`; the API sends 40
// unless the caller names another). So the card hands back those k — log-
// probability, id and the RAW score the draw divides by the temperature —
// and the host draws exactly as it would from the whole row. Before, a
// sampled reply read every row back whole (608 KB at 152K) and ranked it on
// the host: 34-43 ms a flow call against 25-29 greedy on the TH↔AU link
// (2026-10-23, 1.5B Q8_0 drafter, temperature 0.7).
//
// The selection is a radix select, the shape of torch.topk
// (aten/src/ATen/native/cuda/SortingRadixSelect.cuh: the same order-keeping
// key, alternating count buffers): the k-th largest key is found one bit a
// pass, MSB first, by counting the keys at or above each candidate; then
// every key above it is taken, and its ties fill the rest — ANY of them, as
// the host's `select_top_k_candidates` takes any.
//
// Pass 1: grid (chunks, rows), TOPK_THREADS a block, a chunk at most
// TOPK_THREADS * TOPK_PER_THREAD scores. Block (c, r) writes TOPK_PARTIAL_WIDTH
// floats: max, sum of exp(x - max), the bad flag, then KMAX scaled scores and
// KMAX ids bit-cast — the chunk's k likeliest in no order, empty slots
// (-inf, INT_MAX).
// Pass 2: grid rows, TOPK_THREADS a block, chunks * k <= TOPK_THREADS *
// TOPK_MERGE_PER_THREAD. Writes TOPK_OUT_WIDTH floats a row: the log-sum-exp,
// KMAX log-probabilities likeliest first (a tie to the smaller id), KMAX ids as
// exact floats, KMAX raw scores; past the row's finite scores -inf, 0, -inf.
// A NaN or +inf, or nothing finite, writes a NaN log-sum-exp.
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

// ---------------------------------------------------------------------------
// A row's top k for a coupled draw (see the header).

#define KMAX 64
#define TOPK_THREADS 256
#define TOPK_WARPS (TOPK_THREADS / 32)
#define TOPK_PER_THREAD 16
#define TOPK_MERGE_PER_THREAD 32
#define TOPK_PARTIAL_WIDTH (3 + 2 * KMAX)
#define TOPK_OUT_WIDTH (1 + 3 * KMAX)

// A finite float's order-keeping key — torch's `TopKTypeConfig<float>::convert`:
// the higher score, the higher key. 0 is no finite float's key; it marks an
// empty place (absent, -inf).
__device__ __forceinline__ unsigned order_key(float x) {
    const unsigned u = __float_as_uint(x);
    return u ^ ((u & 0x80000000u) ? 0xffffffffu : 0x80000000u);
}

__device__ __forceinline__ float from_order_key(unsigned key) {
    return __uint_as_float(key ^ ((key & 0x80000000u) ? 0x80000000u : 0xffffffffu));
}

// The block's sum of `n`, answered to every thread. `buf` holds TOPK_WARPS
// ints; consecutive calls alternate between two such buffers, so one barrier a
// call is enough: a call's writes come after the previous call's barrier, which
// every thread passed only once it had read the buffer before that.
__device__ __forceinline__ int block_count(int n, int *buf) {
    n = (int)__reduce_add_sync(0xffffffffu, (unsigned)n);
    if ((threadIdx.x & 31) == 0) {
        buf[threadIdx.x >> 5] = n;
    }
    __syncthreads();
    int total = 0;
#pragma unroll
    for (int w = 0; w < TOPK_WARPS; ++w) {
        total += buf[w];
    }
    return total;
}

// The k-th largest of every thread's keys, or 0 where fewer than k are
// present: the largest T with at least k keys >= T, built a bit at a time.
// `counts` holds 2 * TOPK_WARPS ints.
template <int E>
__device__ unsigned kth_largest_key(const unsigned (&keys)[E], int k, int *counts) {
    unsigned prefix = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned want = prefix | (1u << bit);
        int n = 0;
#pragma unroll
        for (int j = 0; j < E; ++j) {
            n += keys[j] >= want;
        }
        if (block_count(n, counts + (bit & 1) * TOPK_WARPS) >= k) {
            prefix = want;
        }
    }
    return prefix;
}

// The block's k largest keys into `vals` / `ids` (slots [0, k), in no order):
// every key above the k-th, then the k-th's ties until k are placed. Answers
// how many were placed (fewer than k only where fewer keys are present). Every
// thread must call it; `fill` is a shared int.
template <int E>
__device__ int take_top_k(const unsigned (&keys)[E], const int (&key_ids)[E], int k,
                          int *counts, int *fill, float *vals, int *ids) {
    const unsigned kth = kth_largest_key(keys, k, counts);
    if (threadIdx.x == 0) {
        *fill = 0;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < E; ++j) {
        if (keys[j] > kth) {
            const int slot = atomicAdd(fill, 1);
            vals[slot] = from_order_key(keys[j]);
            ids[slot] = key_ids[j];
        }
    }
    __syncthreads();
    if (kth != 0) {
#pragma unroll
        for (int j = 0; j < E; ++j) {
            if (keys[j] == kth) {
                const int slot = atomicAdd(fill, 1);
                if (slot < k) {
                    vals[slot] = from_order_key(keys[j]);
                    ids[slot] = key_ids[j];
                }
            }
        }
    }
    __syncthreads();
    return min(*fill, k);
}

extern "C" __global__ void tree_topk_partials(
    const float *__restrict__ logits,
    float *__restrict__ partials,
    const int vocab,
    const int chunks,
    const float inv_t,
    const int k
) {
    __shared__ float lse_shared[3 * TOPK_WARPS + 2];
    __shared__ int counts[2 * TOPK_WARPS];
    __shared__ int fill;
    __shared__ float vals[KMAX];
    __shared__ int ids[KMAX];
    const int chunk = blockIdx.x;
    const int row = blockIdx.y;
    const int chunk_len = (vocab + chunks - 1) / chunks;
    const int start = chunk * chunk_len;
    const int end = min(start + chunk_len, vocab);
    const float *scores = logits + (size_t)row * vocab;

    float m = -CUDART_INF_F;
    float s = 0.0f;
    int bad = 0;
    unsigned keys[TOPK_PER_THREAD];
    int key_ids[TOPK_PER_THREAD];
#pragma unroll
    for (int j = 0; j < TOPK_PER_THREAD; ++j) {
        const int i = start + (int)threadIdx.x + j * TOPK_THREADS;
        keys[j] = 0;
        key_ids[j] = i;
        if (i < end) {
            const float x = scores[i] * inv_t;
            if (isnan(x) || x == CUDART_INF_F) {
                bad = 1;
            } else if (x != -CUDART_INF_F) {
                if (x > m) {
                    s = s * expf(m - x) + 1.0f;
                    m = x;
                } else {
                    s += expf(x - m);
                }
                keys[j] = order_key(x);
            }
        }
    }
    block_lse(m, s, bad, lse_shared);
    float *dst = partials + ((size_t)row * chunks + chunk) * TOPK_PARTIAL_WIDTH;
    if (threadIdx.x == 0) {
        dst[0] = m;
        dst[1] = s;
        dst[2] = (float)bad;
    }
    const int placed = take_top_k(keys, key_ids, k, counts, &fill, vals, ids);
    for (int slot = threadIdx.x; slot < KMAX; slot += TOPK_THREADS) {
        const bool held = slot < placed;
        dst[3 + slot] = held ? vals[slot] : -CUDART_INF_F;
        dst[3 + KMAX + slot] = __int_as_float(held ? ids[slot] : INT_MAX);
    }
}

extern "C" __global__ void tree_topk_merge(
    const float *__restrict__ partials,
    const float *__restrict__ logits,
    float *__restrict__ out,
    const int vocab,
    const int chunks,
    const int k
) {
    __shared__ float lse_shared[3 * TOPK_WARPS + 2];
    __shared__ int counts[2 * TOPK_WARPS];
    __shared__ int fill;
    __shared__ float vals[KMAX];
    __shared__ int ids[KMAX];
    __shared__ float row_lse;
    const int row = blockIdx.x;
    const float *row_partials = partials + (size_t)row * chunks * TOPK_PARTIAL_WIDTH;

    // The row's log-sum-exp, from every chunk's (max, sum); chunks <= the
    // block's width (the host bounds it).
    float m = -CUDART_INF_F;
    float s = 0.0f;
    int bad = 0;
    if ((int)threadIdx.x < chunks) {
        const float *src = row_partials + (size_t)threadIdx.x * TOPK_PARTIAL_WIDTH;
        m = src[0];
        s = src[1];
        bad = src[2] != 0.0f;
    }
    block_lse(m, s, bad, lse_shared);
    if (threadIdx.x == 0) {
        row_lse = (bad || m == -CUDART_INF_F) ? CUDART_NAN_F : m + logf(s);
    }

    // Every chunk's first k places are the candidates.
    unsigned keys[TOPK_MERGE_PER_THREAD];
    int key_ids[TOPK_MERGE_PER_THREAD];
    const int candidates = chunks * k;
#pragma unroll
    for (int j = 0; j < TOPK_MERGE_PER_THREAD; ++j) {
        const int e = (int)threadIdx.x + j * TOPK_THREADS;
        keys[j] = 0;
        key_ids[j] = INT_MAX;
        if (e < candidates) {
            const float *src = row_partials + (size_t)(e / k) * TOPK_PARTIAL_WIDTH;
            const int slot = e % k;
            const int id = __float_as_int(src[3 + KMAX + slot]);
            if (id != INT_MAX) {
                keys[j] = order_key(src[3 + slot]);
                key_ids[j] = id;
            }
        }
    }
    const int placed = take_top_k(keys, key_ids, k, counts, &fill, vals, ids);

    // Each placed token's rank among the placed: the higher score first, a tie
    // to the smaller id (ids are distinct, so the ranks are too).
    float *dst = out + (size_t)row * TOPK_OUT_WIDTH;
    const float lse = row_lse;
    if ((int)threadIdx.x < placed) {
        const float v = vals[threadIdx.x];
        const int id = ids[threadIdx.x];
        int rank = 0;
        for (int u = 0; u < placed; ++u) {
            rank += ranks_before(vals[u], ids[u], v, id);
        }
        dst[1 + rank] = v - lse;
        dst[1 + KMAX + rank] = (float)id;
        dst[1 + 2 * KMAX + rank] = logits[(size_t)row * vocab + id];
    }
    for (int slot = placed + (int)threadIdx.x; slot < KMAX; slot += TOPK_THREADS) {
        dst[1 + slot] = -CUDART_INF_F;
        dst[1 + KMAX + slot] = 0.0f;
        dst[1 + 2 * KMAX + slot] = -CUDART_INF_F;
    }
    if (threadIdx.x == 0) {
        dst[0] = lse;
    }
}
