// SwarmLLM fused decode kernels.
//
// WHY THIS FILE EXISTS
// --------------------
// Decode on the GPU is bound by how many times the CPU thread talks to the
// driver, not by memory bandwidth — measured at 1,085 submissions and 17.6 of
// 23.0 ms/token with the card at 52%.
// → `docs/invariants/inference.md` § "A decode token is bound by GPU
//   submission COUNT, not bandwidth".
//
// So every candle op that a layer composes out of two kernels costs a launch,
// an allocation and a free that a single kernel would not. These are the
// fusions that remove such a pair.
//
// HOW IT GETS INTO THE BINARY
// ---------------------------
// `build.rs` compiles this to PTX with `nvcc --ptx` when the `candle-cuda`
// feature is on, and the PTX is `include_str!`d and handed to candle's
// `CudaDevice::get_or_load_custom_func`. That path had no callers upstream and
// is the whole reason no fifth crate had to be vendored: `candle-kernels` is a
// registry crate, so its `.cu` files are not ours to extend.
//
// ⚠ **Built WITHOUT `-use_fast_math`, deliberately.** Each kernel here has a
// composed candle equivalent that it must agree with BIT FOR BIT, so that an
// A/B between them is a test of the submission count and nothing else. Fast
// math would silently make every reply a little different and turn a
// correctness regression into a judgement call.

#include <cuda_runtime.h>

// silu(gate) * up, the SwiGLU tail of every dense FFN, as ONE kernel.
//
// candle composes it as `usilu_f32` then `bmul_f32` — two launches, two
// allocations and two frees per layer, for an elementwise pass over a few
// KB while decoding.
//
// ⚠ The expression is candle's `silu_fwd` (`x / (1 + expf(-x))`, see
// `candle-kernels/src/unary.cu`) followed by candle's `bmul` (`a * b`), in
// that order and with those exact operations. That is not stylistic: it is
// what makes the fused result bit-identical to the composed one, which
// `cuda_silu_mul_is_bit_identical_to_the_composed_path` asserts. There is no
// fused multiply-add to contract here — an add, a divide, then a multiply —
// so `-fmad=true` (nvcc's default) cannot change the answer either.
//
// The grid-stride loop assigns every element it owns, so the output buffer may
// come from `alloc_fully_overwritten` and skip its zero-fill.
extern "C" __global__ void silu_mul_f32(
    const size_t numel,
    const float *gate,
    const float *up,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        const float g = gate[i];
        out[i] = (g / (1.0f + expf(-g))) * up[i];
    }
}

// (a + b) and rms_norm(a + b) * alpha, as ONE kernel with TWO outputs.
//
// Every dense layer hands its residual stream through this pattern twice —
// `attn + residual` then the FFN norm, and `ffn + residual` then the NEXT
// layer's attention norm (or the final norm) — and candle composes each as
// `badd_f32` then `rmsnorm_f32`: two launches, and an allocation and a free for
// the sum. Both results are needed: the sum IS the next residual, and the
// normed value feeds the projections. candle's `CustomOp` returns one storage,
// so the caller allocates ONE 2N buffer and passes its two halves here;
// `inference::residual_norm` splits it back into two zero-copy views.
//
// ⚠ **This is candle's `rmsnorm` from `candle-kernels/src/reduce.cu`, statement
// for statement** — the same per-thread strided accumulation, the same warp
// reduction (`__shfl_xor_sync` over masks 16..1), the same shared-memory second
// stage when `block_size > 32`, the same `rsqrtf(mean + eps)` and the same
// `(scale * x) * alpha` order. The caller picks `block_size` by candle's rule
// (32 below 1024 columns, else 1024) and launches one block per row, as candle
// does. The only difference is where `x` comes from: candle loads what
// `badd_f32` stored (`x + y`, one IEEE add), and this computes the same add in
// a register. FMA contraction cannot fuse an add INTO the multiply that follows
// it, so `xi` is the same rounded value and `tmp += xi * xi` contracts exactly
// as candle's does. That is what makes the result BIT-identical, which
// `cuda_add_rms_norm_is_bit_identical_to_the_composed_path` asserts.
//
// Both loops assign every element of their row, so the 2N output may come from
// `alloc_fully_overwritten`.
static __device__ __forceinline__ float fd_warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

extern "C" __global__ void add_rmsnorm_f32(
    const float *a,
    const float *b,
    const float *alpha,
    float *normed,
    float *sum,
    const int ncols,
    const int block_size,
    const float eps
) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = a[row*ncols + col] + b[row*ncols + col];
        sum[row*ncols + col] = xi;
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = fd_warp_reduce_sum(tmp);
    if (block_size > 32) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / 32;
        int lane_id = threadIdx.x % 32;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = fd_warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        float al = alpha[col];
        normed[row*ncols + col] = scale * sum[row*ncols + col] * al;
    }
}

// Partial RoPE — rotate the leading `rope_dim` of every head, copy the rest —
// as ONE kernel.
//
// A model whose rotary width is narrower than its heads (Phi-4-mini: 96 of 128;
// GLM-4: 64 of 128; Qwen 3.5's attention layers: 64 of 256) was composed as
// `ucopy` (the rotated part made contiguous), `rope`, `ucopy` (the pass-through
// part) and a `cat` of two `copy2d`s — five launches for Q and five for K where
// a full-width model makes one each. llama.cpp's `rope_neox` / `rope_norm`
// (`ggml/src/ggml-cuda/rope.cu`, master 4b1a27f) copy every channel at or past
// `n_dims` inside the rope kernel itself; this is the same shape.
//
// ⚠ **Bit-identical to the composition.** The rotated pair is candle's `rope`
// (contiguous halves: element j pairs with j + rope_dim/2) or `ropei`
// (interleaved: 2j pairs with 2j+1) from `candle-kernels/src/reduce.cu`,
// expression for expression, with the same cos/sin index (position × half + j);
// the pass-through is a plain copy, as `ucopy` and `copy2d` are.
// `cuda_partial_rope_is_bit_identical_to_the_composed_path` pins it.
//
// LAYOUT: `src` and `dst` are `rows` contiguous heads of `head_dim` (a
// `[b, heads, seq, head_dim]` tensor, so row r is at position r % seq); `cos`
// and `sin` are `[seq, rope_dim / 2]`. Every element of `dst` is assigned, so it
// may come from `alloc_fully_overwritten`.
extern "C" __global__ void rope_partial_f32(
    const float *src,
    const float *cos,
    const float *sin,
    float *dst,
    const unsigned int rows,
    const unsigned int seq,
    const unsigned int head_dim,
    const unsigned int rope_dim,
    const int interleaved
) {
    const unsigned int half = rope_dim / 2;
    const unsigned int per_row = half + (head_dim - rope_dim);
    const size_t total = (size_t)rows * per_row;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        const size_t r = idx / per_row;
        const unsigned int j = (unsigned int)(idx - r * per_row);
        const size_t head = r * head_dim;
        if (j < half) {
            const size_t i_cs = (r % seq) * half + j;
            const float c = cos[i_cs];
            const float s = sin[i_cs];
            const size_t i1 = interleaved ? head + 2 * j : head + j;
            const size_t i2 = interleaved ? i1 + 1 : i1 + half;
            dst[i1] = src[i1] * c - src[i2] * s;
            dst[i2] = src[i1] * s + src[i2] * c;
        } else {
            const size_t k = head + rope_dim + (j - half);
            dst[k] = src[k];
        }
    }
}

// A projection's output to the heads-first layout attention reads, with its
// bias added and RoPE applied on the way — ONE kernel for what candle composes
// as `badd` (the bias), `ucopy` (the `transpose(1, 2).contiguous()` to
// `[b, heads, seq, head_dim]`) and `rope` / `ropei` (or `rope_partial_f32`):
// three launches for Q, three for K, two for V, each a pass over the tensor
// and an allocation. TensorRT-LLM's `addFusedQKVBiasTranspose` is the same
// shape (bias, rotary, transpose in one pass); llama.cpp fuses its rope with
// the cache's `set_rows` for the same reason.
//
// ⚠ **Bit-identical to the composition.** The bias is candle's `badd`
// (`x + b`); the rotated pair is candle's `rope` (contiguous halves: element j
// pairs with j + rope_dim/2) or `ropei` (interleaved: 2j with 2j+1) from
// `candle-kernels/src/reduce.cu`, expression for expression, over the biased
// values, with the same cos/sin index (position × half + j); every other
// channel is copied (biased). `cuda_heads_first_is_bit_identical_to_the_composed_path` pins it.
//
// LAYOUT: `src` is `[b, seq, heads, head_dim]` contiguous (a projection's
// `[b, seq, heads * head_dim]`); `bias` is `[heads * head_dim]` or null;
// `cos` and `sin` are `[seq, rope_dim / 2]` (unread when `rope_dim` is 0);
// `dst` is `[b, heads, seq, head_dim]`. Every element of `dst` is assigned, so
// it may come from `alloc_fully_overwritten`.
extern "C" __global__ void heads_first_f32(
    const float *src,
    const float *bias,
    const float *cos,
    const float *sin,
    float *dst,
    const unsigned int b,
    const unsigned int seq,
    const unsigned int heads,
    const unsigned int head_dim,
    const unsigned int rope_dim,
    const int interleaved
) {
    const unsigned int half = rope_dim / 2;
    const unsigned int per_row = half + (head_dim - rope_dim);
    const size_t total = (size_t)b * heads * seq * per_row;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)blockDim.x * gridDim.x) {
        // `r` is a row of `dst`: (batch, head, position).
        const size_t r = idx / per_row;
        const unsigned int j = (unsigned int)(idx - r * per_row);
        const unsigned int t = (unsigned int)(r % seq);
        const size_t bh = r / seq;
        const unsigned int h = (unsigned int)(bh % heads);
        const size_t bi = bh / heads;
        const float *in = src + ((bi * seq + t) * heads + h) * head_dim;
        const float *bias_h = bias == nullptr ? nullptr : bias + (size_t)h * head_dim;
        float *out = dst + r * head_dim;
        if (j < half) {
            const size_t i_cs = (size_t)t * half + j;
            const float c = cos[i_cs];
            const float s = sin[i_cs];
            const unsigned int i1 = interleaved ? 2 * j : j;
            const unsigned int i2 = interleaved ? i1 + 1 : i1 + half;
            const float x1 = bias_h == nullptr ? in[i1] : in[i1] + bias_h[i1];
            const float x2 = bias_h == nullptr ? in[i2] : in[i2] + bias_h[i2];
            out[i1] = x1 * c - x2 * s;
            out[i2] = x1 * s + x2 * c;
        } else {
            const unsigned int k = rope_dim + (j - half);
            out[k] = bias_h == nullptr ? in[k] : in[k] + bias_h[k];
        }
    }
}
