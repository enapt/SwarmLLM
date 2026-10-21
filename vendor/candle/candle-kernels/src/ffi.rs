use core::ffi::c_void;
#[allow(dead_code)]
extern "C" {
    // for unquntized models
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void,      // device pointer [size_m, size_n]
        expert_counts: *mut i32,  // pre-allocated buffer [num_experts]
        expert_offsets: *mut i32, // pre-allocated buffer [num_experts + 1]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        is_prefill: bool,
        stream: i64,
    );

    pub fn moe_gemm_gguf(
        input: *const f32,      // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        gguf_dtype: i32, // Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5, and (SwarmLLM patch) Q4_0: 15, Q4_1: 16, Q5_0: 17, Q5_1: 18  (for weights)
        stream: i64,
    );

    pub fn moe_gemm_gguf_prefill(
        input: *const c_void, // input [size_m, size_k]
        weights: *const u8,   // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,   //must be host ptr
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // float output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        input_dtype: i32, // 0=f16, 1=bf16 (for inputs)
        gguf_dtype: i32,  //Q8_0: 0, Q4K: 1, Q2K: 2, Q3k: 3,  Q5K: 4, Q6K: 5, and (SwarmLLM patch) Q4_0: 15, Q4_1: 16, Q5_0: 17, Q5_1: 18  (for weights)
        stream: i64,
    );
}

// SwarmLLM patch: llama.cpp's int8 tensor-core quantized matmul, built from
// `src/ggml_mmq` (see `swarm_mmq.cuh` there). `ggml_type` is the GGUF type id;
// `stream` the caller's `CUstream`. Every status is `swarm_mmq_status`: 0 ok,
// 1 unsupported, 2 fixup buffer too small, 3 CUDA error, 4 no tile fits.
#[allow(dead_code)]
extern "C" {
    /// 1 where the kernel runs on the current device (int8 tensor cores and
    /// 48 KiB of shared memory a block).
    pub fn swarm_mmq_available() -> i32;

    /// The activation scale layout a weight type reads (0/1/2), -1 if the
    /// type is not covered. Equal layouts can share a quantized activation.
    pub fn swarm_mmq_q8_1_layout(ggml_type: i32) -> i32;

    /// Bytes of a quantized activation of `ncols_y` rows of `ncols_x` values.
    pub fn swarm_mmq_q8_1_bytes(ncols_x: i64, ncols_y: i64) -> usize;

    pub fn swarm_mmq_quantize(
        y: *const f32,
        vy: *mut c_void,
        ggml_type: i32,
        ncols_x: i64,
        ncols_y: i64,
        stride_y: i64,
        stream: i64,
    ) -> i32;

    /// `dst[ncols_y][nrows_x] = vy · xᵀ`. `launch == 0` only sets `need` (the
    /// floats of fixup buffer the launch will want).
    pub fn swarm_mmq_matmul(
        ggml_type: i32,
        x: *const c_void,
        vy: *const c_void,
        dst: *mut f32,
        fixup: *mut f32,
        fixup_floats: usize,
        need: *mut usize,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        launch: i32,
        stream: i64,
    ) -> i32;
}
