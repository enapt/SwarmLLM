// Single-position (decode) attention on the card, straight over the KV cache —
// f32, or the half cache a card keeps (`decode_attn_f16kv`).
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
//   decode_attn_f32 / _f16kv grid (batch * n_kv_head, n_chunks), DA_THREADS
//   decode_attn_combine_f32  grid (batch * n_head), at every length (see `partial`)
//   decode_attn_combine_staged_f32  grid (batch * n_head, ⌈d / 128⌉): the same
//                            merge bit for bit, its runs' weights staged (default)
// One block holds ONE KV head and EVERY query head of its group, so each K and
// V row of its chunk is read once for the group — the rule the CPU kernel
// (`inference::decode_attn`) learned the hard way (#119: once per query head
// was 7x the traffic on Qwen2.5-7B).
//
// The chunk is fixed for a cache length (`decode_attn::chunk_for`), so these
// kernels' summation order — and the result — depends on the cache length and
// the head layout only, never on the card or the scheduling. (The tensor-core
// body below adds inside `mma`, in an order PTX does not fix: its bits may
// differ between card generations; which kernel runs never does.) It is not bit-identical to the matmul path it
// replaces (a different summation order): replies are judged against
// llama.cpp, as for the tensor-core prompt pass. The chunk is an ARGUMENT
// (16, 32 or `DA_CHUNK` = 64, the most shared memory is sized for), chosen by
// the caller from the group count and the cache length only
// (`decode_attn::chunk_for`): a short cache needs more blocks than one per KV
// head (TinyLlama at 40 positions was 4 blocks of 64 on a 40-SM card), a long
// one has them already and pays for each extra chunk in the merge (FUTURE_WORK
// #309). One kernel at every chunk, so a captured step only updates its
// arguments as the chunk grows with the cache.
//
// A TREE of guesses (#134) runs through the same kernel: each of its `rows`
// query rows is a batch element of its own over the SAME cache (the K/V batch
// is `batch / rows`), and reads a key only where its visibility says so —
// every position before `tree_base` (the confirmed context), and the tree's
// own rows its ancestors and itself, one bit each (`vis`, `vis_words` words a
// row; `draft_tree::RowVisibility`). FlashInfer's packed custom mask, the shape
// SGLang's and vLLM's tree drafts run under CUDA graphs. A masked score is −∞,
// and a chunk whose every score is masked contributes nothing (the combine
// skips its −∞ max). Without a tree (`vis` null, `rows` 1) every line reads
// what it read before, bit for bit.
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
// out:     [b, n_head, d] — written here only when `partial` is null (ONE
//          chunk, written directly: the A/B arm `SWARMLLM_DECODE_ATTN_COMBINE=when-needed`)
// partial: [b * n_kv_head, n_chunks, n_rep, d + 2] = (max, Σexp, Σexp·v) per
//          chunk and query head — written whenever it is passed, one chunk
//          included: the combine then runs at EVERY cache length, so a captured
//          decode step keeps the same kernels as the cache grows past a chunk
//          (`decode_attn::combine_every_length`). One chunk merged is
//          bit-identical to the direct write (its weight is expf(0) = 1).
// softcap: Gemma-2's tanh cap on the scaled score, 0 = none.
// rows: query rows a K/V batch serves (a tree's rows; 1 = a decode step) — q,
//       out and partial are indexed by `batch * rows + row`, K/V by `batch`.
// vis:  null, or `[rows, vis_words]` bits: row t reads cache position p where
//       p < tree_base, or bit (p - tree_base) of its words is set. The caller
//       keeps s_len - tree_base <= 32 * vis_words.
// Requires d % 32 == 0, d <= DA_MAX_D, n_rep <= DA_MAX_REP (checked by the caller).
//
// `KV` is the cache's element type: `float`, or `__half` for the half cache a
// card keeps (`layers::KvStorage::F16`, FUTURE_WORK #194). Each element is
// widened to f32 as it is read and everything after that — the products, the
// softmax, the sums — is f32 as before, so the half cache changes what is
// stored and nothing about how it is summed. llama.cpp's vector kernel reads
// its default f16 cache the same way.
static __device__ __forceinline__ float da_load(const float *p) { return *p; }
static __device__ __forceinline__ float da_load(const __half *p) { return __half2float(*p); }

template <typename KV>
static __device__ __forceinline__ void decode_attn_body(
    const float *__restrict__ q,
    const KV *__restrict__ k,
    const KV *__restrict__ v,
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
    const float softcap,
    const int chunk,
    const int rows,
    const unsigned int *__restrict__ vis,
    const int vis_words,
    const int tree_base
) {
    __shared__ float q_sh[DA_MAX_REP * DA_MAX_D];
    __shared__ float sc[DA_MAX_REP * DA_CHUNK];
    __shared__ float red_m[DA_MAX_REP];
    __shared__ float red_l[DA_MAX_REP];

    const int g = blockIdx.x;
    // `bi` numbers query rows (a batch's tree rows one after another); the
    // cache is the batch's.
    const int bi = g / n_kv_head;
    const int h = g % n_kv_head;
    const int kv_bi = bi / rows;
    const unsigned int *vis_row = vis == nullptr ? nullptr : vis + (long long)(bi % rows) * vis_words;
    const int c = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int s0 = c * chunk;
    const int n = min(chunk, s_len - s0);
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

    const KV *kb = k + kv_bi * k_sb + h * k_sh + (long long)s0 * d;
    const KV *vb = v + kv_bi * v_sb + h * v_sh + (long long)s0 * d;

    // Scores: one warp per position; its K row is read once, for every head
    // of the group. A position this row does not see is −∞ and its K row is
    // never read (the test is the same for the whole warp).
    for (int p = warp; p < n; p += DA_WARPS) {
        const int t = s0 + p - tree_base;
        if (vis_row != nullptr && t >= 0 && !((vis_row[t >> 5] >> (t & 31)) & 1u)) {
            for (int r = lane; r < n_rep; r += 32) {
                sc[r * DA_CHUNK + p] = -INFINITY;
            }
            continue;
        }
        float kr[DA_MAX_D / 32];
#pragma unroll
        for (int i = 0; i < DA_MAX_D / 32; i++) {
            kr[i] = i < per_lane ? da_load(&kb[(long long)p * d + lane + 32 * i]) : 0.0f;
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
            // A chunk this row sees nothing of: no weight, not expf(NaN).
            const float e = m == -INFINITY ? 0.0f : expf(sc[r * DA_CHUNK + p] - m);
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
            const float vv = da_load(&vb[(long long)p * d + j]);
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
                if (partial == nullptr) {
                    out[((long long)g * n_rep + r) * d + j] = red_l[r] == 0.0f ? 0.0f : acc[r] / red_l[r];
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

#define DA_ENTRY(NAME, KV)                                                      \
    extern "C" __global__ void NAME(                                            \
        const float *__restrict__ q, const KV *__restrict__ k,                  \
        const KV *__restrict__ v, float *__restrict__ out,                      \
        float *__restrict__ partial, const int n_kv_head, const int n_rep,      \
        const int d, const int s_len, const long long k_sb,                     \
        const long long k_sh, const long long v_sb, const long long v_sh,       \
        const float scale, const float softcap, const int chunk,               \
        const int rows, const unsigned int *__restrict__ vis,                   \
        const int vis_words, const int tree_base) {                             \
        decode_attn_body<KV>(q, k, v, out, partial, n_kv_head, n_rep, d, s_len, \
                             k_sb, k_sh, v_sb, v_sh, scale, softcap, chunk,     \
                             rows, vis, vis_words, tree_base);                  \
    }

// The f32 cache (the processor's layout, and a card's where the half cache does
// not apply) and the half cache. Same arguments; only the cache pointers' type
// differs.
DA_ENTRY(decode_attn_f32, float)
DA_ENTRY(decode_attn_f16kv, __half)

// ONE WARP A RUN OF POSITIONS, THE GROUP'S QUERY HEADS ON THE TENSOR CORES
// (2026-11-25, FUTURE_WORK #309)
// -------------------------------------------------------------------------
// The kernels above spend a warp on each position: one K row read, then a
// shuffle reduction per query head before the next row, and every block pays
// the query's trip through shared memory and three barriers for 16-64
// positions. On the RTX 3070 Laptop that is 42-45 µs a layer of a 7B at ~960
// positions where llama.cpp's decode takes 13.7 (+3.0 fixup), and two bodies
// that kept more loads in flight (10-19, 11-25) did not close it. This body:
// 12.5 µs there, and the 7B's decode from a 4,096-token prompt 48.5-49.2 →
// 56.7-57.9 tok/s (`docs/invariants/inference.md` § "A decode step over the
// half cache runs on the tensor cores").
//
// Here one warp owns a run of `chunk` positions of one KV head and every query
// head of its group, sixteen positions a step — up to 128 dims with no shared
// memory and no barrier (at 256 the group's query is staged once a block in
// shared memory, `DaMmaQuery`): the query heads are the 16 rows of `mma.m16n8k16` (a group of up to
// 16; rows past `n_rep` are zeros and never written), so
//   S^T[head][pos] = Q[head][dim] · K^T  — K's rows are B's columns as they lie
//                    (`.col`), read as 32-bit pairs straight from the cache;
//   O^T[head][dim] += P^T[head][pos] · V — the two score tiles' accumulators
//                    ARE P^T's A fragment (FlashAttention-2's register reuse),
//                    V's pairs gathered from two rows each.
// An online softmax runs over the run's 16-position steps; the run writes the
// kernels above's `partial` layout — (max, Σexp, Σexp·v) per query head — and
// `decode_attn_combine_f32` merges the runs unchanged.
//
// Precision: the query is scaled and rounded to half, P is rounded to half
// before Σ P·V, the products accumulate in f32 — llama.cpp's MMA kernel does
// the same three. Not bit-identical to the kernels above: judged against
// llama.cpp's replies (`examples/score_against_reference.py`) and pinned
// against f32 attention by
// `decode_attn::tests::the_tensor_core_decode_kernel_matches_f32_attention`.
//
// Requires a half cache, D of 64, 128 or 256, n_rep <= 16, every K/V row starting
// on 4 bytes (the caller checks; `decode_attn::mma_takes`), sm_80+.
// Prior art: llama.cpp `fattn-mma-f16.cuh` (`ncols2`: a KV head's query heads
// are the rows of one tile; stream-K for the cache) and FlashAttention-2
// (Dao, 2023: the C→A fragment reuse, the online softmax).
#define DA_MMA_WARPS 4

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define DA_MMA_AVAILABLE 1
#endif

// The helpers exist only where the body below uses them: a PTX below sm_80
// builds the entries as a `__trap()` and would otherwise warn of each unused one.
#ifdef DA_MMA_AVAILABLE
static __device__ __forceinline__ void da_mma_16816(
    float (&c)[4], const unsigned (&a)[4], const unsigned b0, const unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Two floats as a half pair, the first in the low 16 bits (the element with
// the lower index in every mma fragment).
static __device__ __forceinline__ unsigned da_half2_bits(const float lo, const float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned *>(&h);
}

static __device__ __forceinline__ unsigned da_halves_bits(const __half lo, const __half hi) {
    return (unsigned)__half_as_ushort(lo) | ((unsigned)__half_as_ushort(hi) << 16);
}
#endif

// Up to 128 dims a warp keeps the group's query in registers (KS × 4). At
// 256 the query (64), one step's K (64) and all 256 outputs (128) pass a
// thread's registers, so the query is staged ONCE a block in shared memory —
// every warp of a block serves the same KV head — and each k-step's fragment
// read from there: the same half bits, so the same answer. Rows sit
// `D / 2 + 4` words apart: a fragment's eight rows then fall in different
// banks. (Two warps a run, each half the dims, read K twice and gained 1-3 %:
// `docs/invariants/inference.md` § "A decode step over the half cache".)
template <int D>
struct DaMmaQuery {
    static constexpr bool IN_SHARED = D > 128;
    static constexpr int ROW_WORDS = D / 2 + 4;
};

template <int D>
static __device__ __forceinline__ void decode_attn_mma_body(
    const float *__restrict__ q,
    const __half *__restrict__ k,
    const __half *__restrict__ v,
    float *__restrict__ partial,
    const int n_kv_head,
    const int n_rep,
    const int s_len,
    const long long k_sb,
    const long long k_sh,
    const long long v_sb,
    const long long v_sh,
    const float scale,
    const float softcap,
    const int chunk,
    const int n_chunks
) {
#ifdef DA_MMA_AVAILABLE
    constexpr int KS = D / 16;  // k-steps of the score product
    constexpr int NT = D / 8;   // output tiles of Σ P·V
    const int g = blockIdx.x;   // (batch, KV head)
    const int bi = g / n_kv_head;
    const int h = g % n_kv_head;
    const int lane = threadIdx.x & 31;
    const int c = blockIdx.y * DA_MMA_WARPS + (threadIdx.x >> 5);
    const float *qg = q + (long long)g * n_rep * D;
    constexpr bool Q_SHARED = DaMmaQuery<D>::IN_SHARED;
    constexpr int QW = DaMmaQuery<D>::ROW_WORDS;
    // 16 rows of the group's query as half pairs, scaled, rounded to half;
    // rows past `n_rep` zero. Staged before any warp returns: the barrier
    // needs every thread of the block.
    __shared__ unsigned q_shared[Q_SHARED ? 16 * QW : 1];
    if constexpr (Q_SHARED) {
        for (int i = threadIdx.x; i < 16 * (D / 2); i += DA_MMA_WARPS * 32) {
            const int r = i / (D / 2);
            const int w = i % (D / 2);
            q_shared[r * QW + w] = r < n_rep
                ? da_half2_bits(qg[r * D + 2 * w] * scale, qg[r * D + 2 * w + 1] * scale)
                : 0u;
        }
        __syncthreads();
    }
    if (c >= n_chunks) {
        return;
    }
    const int s0 = c * chunk;
    const int n = min(chunk, s_len - s0);
    // mma fragment coordinates: `gid` picks a row (A, C) or a column (B),
    // `tig` a pair of columns (A, C) or rows (B).
    const int gid = lane >> 2;
    const int tig = lane & 3;

    // The query heads as A: rows gid and gid + 8, scaled, rounded to half.
    // Fragment f of k-step ks: row gid + 8 (f & 1), the half pair at column
    // 16 ks + 2 tig + 8 (f >> 1).
    unsigned qa[Q_SHARED ? 1 : KS][4];
    if constexpr (!Q_SHARED) {
#pragma unroll
        for (int ks = 0; ks < KS; ks++) {
#pragma unroll
            for (int f = 0; f < 4; f++) {
                const int r = gid + 8 * (f & 1);
                const int col = ks * 16 + 2 * tig + 8 * (f >> 1);
                qa[ks][f] = r < n_rep
                    ? da_half2_bits(qg[r * D + col] * scale, qg[r * D + col + 1] * scale)
                    : 0u;
            }
        }
    }

    const __half *kb = k + bi * k_sb + h * k_sh + (long long)s0 * D;
    const __half *vb = v + bi * v_sb + h * v_sh + (long long)s0 * D;

    float o[NT][4];
#pragma unroll
    for (int t = 0; t < NT; t++) {
#pragma unroll
        for (int i = 0; i < 4; i++) {
            o[t][i] = 0.0f;
        }
    }
    // Running max and Σexp of rows gid (0) and gid + 8 (1), the same in the
    // row's four threads.
    float m[2] = {-INFINITY, -INFINITY};
    float l[2] = {0.0f, 0.0f};

    for (int p0 = 0; p0 < n; p0 += 16) {
        // Scores of positions p0..p0+15: two 8-position tiles, column gid of
        // tile t is position p0 + 8t + gid — its K row read as half pairs.
        float s[2][4];
        unsigned kf[2][KS][2];
#pragma unroll
        for (int t = 0; t < 2; t++) {
            const int p = p0 + 8 * t + gid;
            const unsigned *krow = reinterpret_cast<const unsigned *>(kb + (long long)p * D);
#pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                kf[t][ks][0] = p < n ? krow[ks * 8 + tig] : 0u;
                kf[t][ks][1] = p < n ? krow[ks * 8 + tig + 4] : 0u;
            }
        }
#pragma unroll
        for (int t = 0; t < 2; t++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                s[t][i] = 0.0f;
            }
        }
        // Each tile adds its k-steps in order, whichever way the query is held.
        if constexpr (Q_SHARED) {
#pragma unroll
            for (int ks = 0; ks < KS; ks++) {
                unsigned a_shared[4];
#pragma unroll
                for (int f = 0; f < 4; f++) {
                    a_shared[f] = q_shared[(gid + 8 * (f & 1)) * QW + ks * 8 + tig + 4 * (f >> 1)];
                }
#pragma unroll
                for (int t = 0; t < 2; t++) {
                    da_mma_16816(s[t], a_shared, kf[t][ks][0], kf[t][ks][1]);
                }
            }
        } else {
#pragma unroll
            for (int t = 0; t < 2; t++) {
#pragma unroll
                for (int ks = 0; ks < KS; ks++) {
                    da_mma_16816(s[t], qa[ks], kf[t][ks][0], kf[t][ks][1]);
                }
            }
        }
        // Accumulator i of tile t: row gid + 8 (i >> 1), position
        // p0 + 8t + 2 tig + (i & 1). Soft cap; positions past the run are −∞.
        float tmax[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int t = 0; t < 2; t++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float x = s[t][i];
                if (softcap > 0.0f) {
                    x = softcap * tanhf(x / softcap);
                }
                x = p0 + 8 * t + 2 * tig + (i & 1) < n ? x : -INFINITY;
                s[t][i] = x;
                tmax[i >> 1] = fmaxf(tmax[i >> 1], x);
            }
        }
        float scale_old[2];
        float tsum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            // A row's four threads differ in the low two lane bits.
            tmax[hr] = fmaxf(tmax[hr], __shfl_xor_sync(0xffffffff, tmax[hr], 1));
            tmax[hr] = fmaxf(tmax[hr], __shfl_xor_sync(0xffffffff, tmax[hr], 2));
            const float mn = fmaxf(m[hr], tmax[hr]);
            // Nothing seen yet keeps a zero sum; past that, the old sums'
            // weight under the new max.
            scale_old[hr] = mn == -INFINITY ? 1.0f : expf(m[hr] - mn);
            m[hr] = mn;
        }
#pragma unroll
        for (int t = 0; t < 2; t++) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float mr = m[i >> 1];
                const float e = mr == -INFINITY ? 0.0f : expf(s[t][i] - mr);
                s[t][i] = e;
                tsum[i >> 1] += e;
            }
        }
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            tsum[hr] += __shfl_xor_sync(0xffffffff, tsum[hr], 1);
            tsum[hr] += __shfl_xor_sync(0xffffffff, tsum[hr], 2);
            l[hr] = l[hr] * scale_old[hr] + tsum[hr];
        }
        // P^T as A: rows gid / gid + 8, positions 2 tig (+1) of tile 0, then
        // of tile 1 — exactly the two score tiles' accumulators.
        const unsigned pa[4] = {
            da_half2_bits(s[0][0], s[0][1]),
            da_half2_bits(s[0][2], s[0][3]),
            da_half2_bits(s[1][0], s[1][1]),
            da_half2_bits(s[1][2], s[1][3]),
        };
        // V as B: column gid of tile t is dim 8t + gid; rows (positions)
        // 2 tig, 2 tig + 1 and 2 tig + 8, 2 tig + 9. A position past the run
        // reads as zero: its P is zero, and 0 × a NaN left in the buffer is not.
        const int pv = p0 + 2 * tig;
        const __half zero = __float2half(0.0f);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const __half *col = vb + 8 * t + gid;
            const unsigned b0 = da_halves_bits(
                pv < n ? col[(long long)pv * D] : zero,
                pv + 1 < n ? col[(long long)(pv + 1) * D] : zero
            );
            const unsigned b1 = da_halves_bits(
                pv + 8 < n ? col[(long long)(pv + 8) * D] : zero,
                pv + 9 < n ? col[(long long)(pv + 9) * D] : zero
            );
            o[t][0] *= scale_old[0];
            o[t][1] *= scale_old[0];
            o[t][2] *= scale_old[1];
            o[t][3] *= scale_old[1];
            da_mma_16816(o[t], pa, b0, b1);
        }
    }

    // Accumulator i of output tile t: row gid + 8 (i >> 1), dim 8t + 2 tig + (i & 1).
#pragma unroll
    for (int hr = 0; hr < 2; hr++) {
        const int r = gid + 8 * hr;
        if (r < n_rep) {
            float *part = partial + (((long long)g * n_chunks + c) * n_rep + r) * (D + 2);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                part[2 + 8 * t + 2 * tig] = o[t][2 * hr];
                part[2 + 8 * t + 2 * tig + 1] = o[t][2 * hr + 1];
            }
            if (tig == 0) {
                part[0] = m[hr];
                part[1] = l[hr];
            }
        }
    }
#else
    __trap();
#endif
}

#define DA_MMA_ENTRY(NAME, D)                                                      \
    extern "C" __global__ void __launch_bounds__(DA_MMA_WARPS * 32) NAME(          \
        const float *__restrict__ q, const __half *__restrict__ k,                 \
        const __half *__restrict__ v, float *__restrict__ partial,                 \
        const int n_kv_head, const int n_rep, const int s_len,                     \
        const long long k_sb, const long long k_sh, const long long v_sb,          \
        const long long v_sh, const float scale, const float softcap,              \
        const int chunk, const int n_chunks) {                                     \
        decode_attn_mma_body<D>(q, k, v, partial, n_kv_head, n_rep, s_len, k_sb,   \
                                k_sh, v_sb, v_sh, scale, softcap, chunk, n_chunks); \
    }

DA_MMA_ENTRY(decode_attn_mma_f16kv_d64, 64)
DA_MMA_ENTRY(decode_attn_mma_f16kv_d128, 128)
// Gemma 2/3, Gemma 4's windowed layers, Qwen 3.5/3.6's attention layers.
DA_MMA_ENTRY(decode_attn_mma_f16kv_d256, 256)

// A TREE's rows TILED over the half cache (#134 (b)): one block stages ONE KV
// head's chunk of K and V in shared memory and serves a tile of the tree's
// query vectors from it — every row of the tree and every query head of the
// group, `rows * n_rep` vectors a KV head, `QT` a block.
//
// The kernel above is one block per (row, KV head, chunk): each row re-reads
// its chunk of the cache and sums each score with a warp's shuffles, so its
// time grows with rows × keys and a 7B's 25 rows at 900 keys took 597 µs
// against the matmul path's 394 (`decode_attn::tree_attention_timing`) — the
// reason trees of more than 8 rows went to that path, which widens the whole
// half cache to f32 and runs two cuBLAS products around a mask and a softmax
// (~0.41 ms a layer of a 24-row near-half forward, 2026-10-22 nsys). FlashInfer
// and SGLang's tree kernels share each block's K/V reads among a tile of query
// rows the same way; this is that tile on CUDA cores, f32 sums, no tensor cores.
//
//   grid (batch * n_kv_head, n_chunks, ceil(rows * n_rep / QT)), 256 threads
//
// Scores: a 16 × 16 thread grid, each thread a (QT/16) × (CH/16) block of
// (query, key) sums over the head dim in order, two dims a step; the staged K
// row is padded to an odd number of words, so the 16 keys a warp reads sit in
// 16 banks. Then each query's max-shifted exp and its sum, a warp a query, as
// the kernel above; then Σ exp·V, a warp's 32 threads over the row's words and
// each thread every eighth query of the tile, in position order as above. It
// writes the kernel above's `partial` layout — (max, Σexp, Σexp·v) per
// (row, KV head, chunk, query head) — so `decode_attn_combine_f32` merges it
// unchanged. A masked score is −∞, a chunk a query sees nothing of has no
// weight; positions past the cache are staged as zeros and masked.
//
// The cache must be the half cache, its rows dense and every row starting on
// 16 bytes (the caller checks the offset and the strides are multiples of 8).
// d % 32 == 0, d <= DMAX. Not bit-identical to the kernel above (each score is
// summed in order where that one sums lanes and shuffles); pinned against the
// matmul path by `cuda_tree_kernel_matches_the_masked_matmul_path`, replies
// judged against llama.cpp.
template <int CH, int QT, int DMAX>
static __device__ __forceinline__ void tree_attn_tiled_body(
    const float *__restrict__ q,
    const __half *__restrict__ k,
    const __half *__restrict__ v,
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
    const float softcap,
    const int rows,
    const unsigned int *__restrict__ vis,
    const int vis_words,
    const int tree_base
) {
    constexpr int THREADS = 256;
    // Words of a staged K/V row: odd, so 16 rows read at one word sit in 16 banks.
    constexpr int KW = DMAX / 2 + 1;
    // Floats of a staged query: two queries a warp reads sit two banks apart.
    constexpr int QS = DMAX + 2;
    // Floats of a score row.
    constexpr int SS = CH + 1;
    constexpr int KM = CH / 16;
    constexpr int QN = QT / 16;
    constexpr int QN2 = QT / 8;
    constexpr int DW = DMAX / 64;
    static_assert(CH % 32 == 0 && QT % 16 == 0 && DMAX % 64 == 0, "tile shape");

    __shared__ float q_sh[QT * QS];
    __shared__ unsigned int kv_sh[CH * KW];
    __shared__ float s_sh[QT * SS];
    __shared__ float m_sh[QT];
    __shared__ float l_sh[QT];

    const int g = blockIdx.x;
    const int kv_bi = g / n_kv_head;
    const int h = g % n_kv_head;
    const int c = blockIdx.y;
    const int n_chunks = gridDim.y;
    const int s0 = c * CH;
    const int n = min(CH, s_len - s0);
    const int q0 = blockIdx.z * QT;
    const int nq = min(QT, rows * n_rep - q0);
    const int n_head = n_kv_head * n_rep;
    const int dw = d >> 1;
    const int tid = threadIdx.x;

    // Every load of the block is issued before the first is waited on: a
    // fixed count a thread, unrolled (a loop bounded by `d` issued one load,
    // waited, stored, and the block sat on memory — 37 µs a block against the
    // few its arithmetic needs, nsys 2026-10-22). K and V go 16 bytes a load
    // (eight halves; the caller checks every row starts on 16 bytes), the
    // queries 4 (their view's offset is not checked), and V waits in registers
    // until the scores are done.
    constexpr int C8 = DMAX / 8;
    constexpr int KV_LOADS = CH * C8 / THREADS;
    constexpr int Q_LOADS = QT * DMAX / THREADS;
    static_assert(KV_LOADS * THREADS == CH * C8 && Q_LOADS * THREADS == QT * DMAX, "staging shape");
    __shared__ long long q_at[QT];

    // Where each query vector of the tile starts; vector `q0 + i` is row
    // (q0 + i) / n_rep, query head h * n_rep + (q0 + i) % n_rep. −1: past the tree's.
    if (tid < QT) {
        long long at = -1;
        if (tid < nq) {
            const int qg = q0 + tid;
            const int row = qg / n_rep;
            const int r = qg - row * n_rep;
            at = ((long long)(kv_bi * rows + row) * n_head + h * n_rep + r) * d;
        }
        q_at[tid] = at;
    }
    const __half *kb = k + kv_bi * k_sb + h * k_sh + (long long)s0 * d;
    const __half *vb = v + kv_bi * v_sb + h * v_sh + (long long)s0 * d;
    const int d8 = d >> 3;
    uint4 kreg[KV_LOADS];
    uint4 vreg[KV_LOADS];
#pragma unroll
    for (int t = 0; t < KV_LOADS; t++) {
        const int i = tid + t * THREADS;
        const int p = i / C8;
        const int c8 = i % C8;
        const bool in = p < n && c8 < d8;
        kreg[t] = in ? reinterpret_cast<const uint4 *>(kb + (long long)p * d)[c8] : make_uint4(0u, 0u, 0u, 0u);
        vreg[t] = in ? reinterpret_cast<const uint4 *>(vb + (long long)p * d)[c8] : make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    float qreg[Q_LOADS];
#pragma unroll
    for (int t = 0; t < Q_LOADS; t++) {
        const int i = tid + t * THREADS;
        const int qi = i / DMAX;
        const int j = i % DMAX;
        const long long at = q_at[qi];
        qreg[t] = at >= 0 && j < d ? q[at + j] : 0.0f;
    }
    // Rows past the cache, and words past `d`, are zeros.
#pragma unroll
    for (int t = 0; t < KV_LOADS; t++) {
        const int i = tid + t * THREADS;
        unsigned int *dst = &kv_sh[(i / C8) * KW + 4 * (i % C8)];
        dst[0] = kreg[t].x;
        dst[1] = kreg[t].y;
        dst[2] = kreg[t].z;
        dst[3] = kreg[t].w;
    }
#pragma unroll
    for (int t = 0; t < Q_LOADS; t++) {
        const int i = tid + t * THREADS;
        q_sh[(i / DMAX) * QS + i % DMAX] = qreg[t];
    }
    __syncthreads();

    {
        const int kx = tid & 15;
        const int qy = tid >> 4;
        float acc[QN][KM];
#pragma unroll
        for (int a = 0; a < QN; a++) {
#pragma unroll
            for (int m = 0; m < KM; m++) {
                acc[a][m] = 0.0f;
            }
        }
        // `dw` is a multiple of 16 (d % 32 == 0).
#pragma unroll 8
        for (int w = 0; w < dw; w++) {
            float2 kk[KM];
#pragma unroll
            for (int m = 0; m < KM; m++) {
                kk[m] = __half22float2(*reinterpret_cast<const __half2 *>(&kv_sh[(kx + 16 * m) * KW + w]));
            }
#pragma unroll
            for (int a = 0; a < QN; a++) {
                const float2 qq = *reinterpret_cast<const float2 *>(&q_sh[(qy + 16 * a) * QS + 2 * w]);
#pragma unroll
                for (int m = 0; m < KM; m++) {
                    acc[a][m] += qq.x * kk[m].x;
                    acc[a][m] += qq.y * kk[m].y;
                }
            }
        }
#pragma unroll
        for (int a = 0; a < QN; a++) {
            const int qi = qy + 16 * a;
            const int row = (q0 + qi) / n_rep;
#pragma unroll
            for (int m = 0; m < KM; m++) {
                const int p = kx + 16 * m;
                float x = acc[a][m] * scale;
                if (softcap > 0.0f) {
                    x = softcap * tanhf(x / softcap);
                }
                bool seen = p < n && qi < nq;
                const int t = s0 + p - tree_base;
                if (seen && vis != nullptr && t >= 0) {
                    seen = (vis[(long long)row * vis_words + (t >> 5)] >> (t & 31)) & 1u;
                }
                s_sh[qi * SS + p] = seen ? x : -INFINITY;
            }
        }
    }
    __syncthreads();

    // V over K's staging (every score is written, so every K read is done),
    // then each query's max-shifted exp and its sum, a warp a query.
#pragma unroll
    for (int t = 0; t < KV_LOADS; t++) {
        const int i = tid + t * THREADS;
        unsigned int *dst = &kv_sh[(i / C8) * KW + 4 * (i % C8)];
        dst[0] = vreg[t].x;
        dst[1] = vreg[t].y;
        dst[2] = vreg[t].z;
        dst[3] = vreg[t].w;
    }
    {
        const int lane = tid & 31;
        const int warp = tid >> 5;
        for (int qi = warp; qi < QT; qi += THREADS / 32) {
            float m = -INFINITY;
            for (int p = lane; p < CH; p += 32) {
                m = fmaxf(m, s_sh[qi * SS + p]);
            }
            m = da_warp_max(m);
            float l = 0.0f;
            for (int p = lane; p < CH; p += 32) {
                const float e = m == -INFINITY ? 0.0f : expf(s_sh[qi * SS + p] - m);
                s_sh[qi * SS + p] = e;
                l += e;
            }
            l = da_warp_sum(l);
            if (lane == 0) {
                m_sh[qi] = m;
                l_sh[qi] = l;
            }
        }
    }
    __syncthreads();

    {
        const int dx = tid & 31;
        const int qy = tid >> 5;
        float2 o[QN2][DW];
#pragma unroll
        for (int a = 0; a < QN2; a++) {
#pragma unroll
            for (int w = 0; w < DW; w++) {
                o[a][w] = make_float2(0.0f, 0.0f);
            }
        }
#pragma unroll 4
        for (int p = 0; p < n; p++) {
            float2 vv[DW];
#pragma unroll
            for (int w = 0; w < DW; w++) {
                const int word = dx + 32 * w;
                vv[w] = word < dw ? __half22float2(*reinterpret_cast<const __half2 *>(&kv_sh[p * KW + word]))
                                  : make_float2(0.0f, 0.0f);
            }
#pragma unroll
            for (int a = 0; a < QN2; a++) {
                const float e = s_sh[(qy + 8 * a) * SS + p];
#pragma unroll
                for (int w = 0; w < DW; w++) {
                    o[a][w].x += e * vv[w].x;
                    o[a][w].y += e * vv[w].y;
                }
            }
        }
#pragma unroll
        for (int a = 0; a < QN2; a++) {
            const int qi = qy + 8 * a;
            if (qi < nq) {
                const int qg = q0 + qi;
                const int row = qg / n_rep;
                const int r = qg - row * n_rep;
                float *part = partial
                    + ((((long long)(kv_bi * rows + row) * n_kv_head + h) * n_chunks + c) * n_rep + r) * (d + 2);
#pragma unroll
                for (int w = 0; w < DW; w++) {
                    const int word = dx + 32 * w;
                    if (word < dw) {
                        part[2 + 2 * word] = o[a][w].x;
                        part[3 + 2 * word] = o[a][w].y;
                    }
                }
                if (dx == 0) {
                    part[0] = m_sh[qi];
                    part[1] = l_sh[qi];
                }
            }
        }
    }
}

#define TA_ENTRY(NAME, CH, QT, DMAX)                                             \
    extern "C" __global__ void __launch_bounds__(256) NAME(                       \
        const float *__restrict__ q, const __half *__restrict__ k,               \
        const __half *__restrict__ v, float *__restrict__ partial,               \
        const int n_kv_head, const int n_rep, const int d, const int s_len,      \
        const long long k_sb, const long long k_sh, const long long v_sb,        \
        const long long v_sh, const float scale, const float softcap,            \
        const int rows, const unsigned int *__restrict__ vis,                    \
        const int vis_words, const int tree_base) {                              \
        tree_attn_tiled_body<CH, QT, DMAX>(q, k, v, partial, n_kv_head, n_rep, d, \
                                           s_len, k_sb, k_sh, v_sb, v_sh, scale, \
                                           softcap, rows, vis, vis_words,        \
                                           tree_base);                           \
    }

// Head dims to 128 in chunks of 64 keys, 32 queries a block (~41 KB of shared
// memory); to 256 in chunks of 32, 16 queries (~35 KB) — both under the 48 KB
// a launch takes without raising its limit (a driver call no capture may make).
// `decode_attn::tiled_chunk` is the chunk each one is launched with.
TA_ENTRY(tree_attn_tiled_f16kv_d128, 64, 32, 128)
TA_ENTRY(tree_attn_tiled_f16kv_d256, 32, 16, 256)

// Merge the chunks: out = Σ_c e^(m_c - M) o_c / Σ_c e^(m_c - M) l_c — the CPU
// kernel's pass 2, in the same order. A tree's rows are batch elements here
// too (`b * rows` of them), and a chunk a row saw nothing of (max −∞) is
// skipped.
//
// A tree whose context flash attention read (`past_runs` > 0, one K/V batch:
// `decode_attn::flash_past`) adds each run's normalised output `o_u` and
// log-sum-exp `L_u` as one more partial: max L_u, sum 1, Σ·v o_u — e^(L_u - M)
// is that run's Σ e^(s - M). flash's layout: query vector `row * n_rep + r` of
// KV head `h` — `past_o` [runs, rows·n_rep, n_kv_head, d] half, `past_lse`
// [runs, n_kv_head, rows·n_rep] — FlashInfer's `merge_state`, the cascade
// that reads a shared prefix once. A run that read no keys (L = +∞) is skipped.
//
// `sinks` (null = none): gpt-oss's learned per-head attention sink, [n_head]
// f32 — one more score with no value, in the scaled-score space: it raises the
// max and adds e^(sink - M) to the denominator only (llama.cpp's
// `soft_max_f32` with `sinks`, `ggml_soft_max_add_sinks`). A row is a query
// head, so `sinks[hh]`.
extern "C" __global__ void decode_attn_combine_f32(
    const float *__restrict__ partial,
    float *__restrict__ out,
    const int n_kv_head,
    const int n_rep,
    const int d,
    const int n_chunks,
    const __half *__restrict__ past_o,
    const float *__restrict__ past_lse,
    const int past_runs,
    const float *__restrict__ sinks
) {
    const int bh = blockIdx.x;
    const int n_head = n_kv_head * n_rep;
    const int bi = bh / n_head;
    const int hh = bh % n_head;
    const int g = bi * n_kv_head + hh / n_rep;
    const int r = hh % n_rep;
    const int row = d + 2;
    // flash's query vector and their count (`gridDim.x / n_head` rows).
    const int sq = bi * n_rep + r;
    const int n_sq = (gridDim.x / n_head) * n_rep;
    const int kvh = hh / n_rep;

    const float sink = sinks ? sinks[hh] : -INFINITY;
    float big_m = sink;
    for (int c = 0; c < n_chunks; c++) {
        big_m = fmaxf(big_m, partial[((long long)(g * n_chunks + c) * n_rep + r) * row]);
    }
    for (int u = 0; u < past_runs; u++) {
        const float m = past_lse[((long long)u * n_kv_head + kvh) * n_sq + sq];
        if (isfinite(m)) {
            big_m = fmaxf(big_m, m);
        }
    }
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float num = 0.0f;
        float den = sinks ? expf(sink - big_m) : 0.0f;
        for (int u = 0; u < past_runs; u++) {
            const float m = past_lse[((long long)u * n_kv_head + kvh) * n_sq + sq];
            if (!isfinite(m)) {
                continue;
            }
            const float w = expf(m - big_m);
            den += w;
            num += w * __half2float(past_o[(((long long)u * n_sq + sq) * n_kv_head + kvh) * d + j]);
        }
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

// THE MERGE WITH THE RUNS' WEIGHTS STAGED (2026-11-25, FUTURE_WORK #309's
// residual). Above, every thread walks the runs itself — the run's max, then
// its sum and output, an `expf` between, ~60 runs at ~1,000 positions in 16s
// — and the grid is one block a query head (28 on a 7B, 8 on Gemma 2 2B), so
// the merge took 9.7 µs a layer at ~928 positions on the 7B and 14.4 at
// ~4,128 on Gemma 2 2B: nearly what the tensor-core attention before it
// costs. Here a block finds the max with its threads reading runs side by
// side, stages each tile of runs' (max, sum) and weight `expf(m - M)` in
// shared memory ONCE, and a thread's walk over its output reads only the
// runs' `Σexp·v`, eight loads in flight (32 were slower: 9.5 against 8.4 µs
// on the 7B, 196 registers); grid.y spreads the dims over
// blocks of `DA_MERGE_TILE`. Each output adds the same products in the same
// order as `decode_attn_combine_f32` (the weight is the same `expf` of the
// same operands; the same runs skipped; `fmaxf`'s max does not depend on
// the order it is taken in), so its answer is that kernel's bit for bit
// (`decode_attn::tests::the_staged_merge_is_the_serial_one_bit_for_bit`).
// `SWARMLLM_DECODE_ATTN_MERGE=serial` keeps the kernel above.
// Prior art: llama.cpp's `flash_attn_combine_results` (fattn-common.cuh)
// stages its (max, sum) pairs in shared memory the same way; its threads
// still take an `expf` a run, over the handful of parallel blocks it splits
// a cache into, where ours merges one run per 16-64 positions.
#define DA_MERGE_TILE 128
#define DA_MERGE_UNROLL 8
extern "C" __global__ void __launch_bounds__(DA_MERGE_TILE) decode_attn_combine_staged_f32(
    const float *__restrict__ partial,
    float *__restrict__ out,
    const int n_kv_head,
    const int n_rep,
    const int d,
    const int n_chunks,
    const __half *__restrict__ past_o,
    const float *__restrict__ past_lse,
    const int past_runs,
    const float *__restrict__ sinks
) {
    __shared__ float tile_m[DA_MERGE_TILE];
    __shared__ float tile_l[DA_MERGE_TILE];
    __shared__ float tile_w[DA_MERGE_TILE];
    __shared__ float warp_max[DA_MERGE_TILE / 32];
    const int bh = blockIdx.x;
    const int n_head = n_kv_head * n_rep;
    const int bi = bh / n_head;
    const int hh = bh % n_head;
    const int g = bi * n_kv_head + hh / n_rep;
    const int r = hh % n_rep;
    const int row = d + 2;
    const int sq = bi * n_rep + r;
    const int n_sq = (gridDim.x / n_head) * n_rep;
    const int kvh = hh / n_rep;
    const int j = blockIdx.y * DA_MERGE_TILE + threadIdx.x;
    const bool mine = j < d;
    // Run c of this query head: `((g * n_chunks + c) * n_rep + r) * row`.
    const float *run0 = partial + ((long long)g * n_chunks * n_rep + r) * row;
    const long long run_stride = (long long)n_rep * row;

    const float sink = sinks ? sinks[hh] : -INFINITY;
    float big_m = sink;
    for (int c = threadIdx.x; c < n_chunks; c += DA_MERGE_TILE) {
        big_m = fmaxf(big_m, run0[c * run_stride]);
    }
    for (int u = threadIdx.x; u < past_runs; u += DA_MERGE_TILE) {
        const float m = past_lse[((long long)u * n_kv_head + kvh) * n_sq + sq];
        if (isfinite(m)) {
            big_m = fmaxf(big_m, m);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        big_m = fmaxf(big_m, __shfl_xor_sync(0xffffffffu, big_m, o));
    }
    if ((threadIdx.x & 31) == 0) {
        warp_max[threadIdx.x >> 5] = big_m;
    }
    __syncthreads();
#pragma unroll
    for (int w = 0; w < DA_MERGE_TILE / 32; w++) {
        big_m = fmaxf(big_m, warp_max[w]);
    }

    float num = 0.0f;
    float den = sinks ? expf(sink - big_m) : 0.0f;
    if (mine) {
        for (int u = 0; u < past_runs; u++) {
            const float m = past_lse[((long long)u * n_kv_head + kvh) * n_sq + sq];
            if (!isfinite(m)) {
                continue;
            }
            const float w = expf(m - big_m);
            den += w;
            num += w * __half2float(past_o[(((long long)u * n_sq + sq) * n_kv_head + kvh) * d + j]);
        }
    }
    for (int t0 = 0; t0 < n_chunks; t0 += DA_MERGE_TILE) {
        const int tn = min(DA_MERGE_TILE, n_chunks - t0);
        // The tile before is read by every thread before it is replaced.
        __syncthreads();
        if (threadIdx.x < tn) {
            const float *part = run0 + (long long)(t0 + threadIdx.x) * run_stride;
            const float m = part[0];
            tile_m[threadIdx.x] = m;
            tile_l[threadIdx.x] = part[1];
            tile_w[threadIdx.x] = m == -INFINITY ? 0.0f : expf(m - big_m);
        }
        __syncthreads();
        if (mine) {
            const float *xs = run0 + (long long)t0 * run_stride + 2 + j;
            for (int c0 = 0; c0 < tn; c0 += DA_MERGE_UNROLL) {
                float x[DA_MERGE_UNROLL];
#pragma unroll
                for (int u = 0; u < DA_MERGE_UNROLL; u++) {
                    x[u] = c0 + u < tn ? xs[(long long)(c0 + u) * run_stride] : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < DA_MERGE_UNROLL; u++) {
                    const int c = c0 + u;
                    // The kernel above skips exactly these: a run that saw
                    // nothing (a NaN max is NOT skipped there either).
                    if (c < tn && tile_m[c] != -INFINITY) {
                        const float w = tile_w[c];
                        den += w * tile_l[c];
                        num += w * x[u];
                    }
                }
            }
        }
    }
    if (mine) {
        out[(long long)bh * d + j] = den > 0.0f ? num / den : 0.0f;
    }
}
