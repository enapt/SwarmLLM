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
