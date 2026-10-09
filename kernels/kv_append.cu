// Write a step's K or V into a half-precision KV cache, converting as it goes.
//
// WHY THIS FILE EXISTS
// --------------------
// On a card the KV cache is stored as f16 (`layers::KvStorage::F16`, FUTURE_WORK
// #194): a third of the memory of the f32 cache plus the f16 flash mirror it
// replaced, and what llama.cpp stores by default. The projections that produce
// K and V are f32, so every append converts. Composed from candle ops that is a
// cast (a launch, an allocation, a free) and then `slice_set` (a launch) — two
// launches per tensor per layer per decoded token where the f32 cache took one,
// and a decode step on a card is bound by how many launches it makes
// (`docs/invariants/inference.md` § "A decode token is bound by GPU submission
// COUNT"). This is the cast and the copy as ONE launch, so the half cache costs
// the step no more submissions than the f32 cache did.
//
// The conversion is `__float2half` — round to nearest even — which is what
// candle's own cast does, so a cache written here holds bitwise the values the
// flash mirror used to hold (`LayerKv::to_bshd_f16`).
//
// LAYOUT
// ------
// `slice_set`'s view of the write: the destination is `rows` rows of
// `dst_row_stride` elements (every dim before the sequence axis collapsed into
// rows, the sequence axis and everything after it into one row), and the source
// fills `cols` elements of each row starting at `dst_offset`. Both are
// contiguous; the caller checks.
//
// Built WITHOUT `-use_fast_math`, like every kernel here (`build.rs`).

#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void kv_write_f16(
    const float *__restrict__ src,
    __half *__restrict__ dst,
    const size_t rows,
    const size_t cols,
    const size_t dst_row_stride,
    const size_t dst_offset
) {
    const size_t total = rows * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        const size_t r = i / cols;
        const size_t c = i - r * cols;
        dst[r * dst_row_stride + dst_offset + c] = __float2half(src[i]);
    }
}
