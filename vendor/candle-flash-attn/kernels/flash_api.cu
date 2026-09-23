#include "kernels.h"
#include "kernel_helpers.h"
#include "flash_fwd_launch_template.h"

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
  FP16_SWITCH(!params.is_bf16, [&] {
      HEADDIM_SWITCH(params.d, [&] {
          BOOL_SWITCH(params.is_causal, Is_causal, [&] {
              run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
          });
      });
  });
}

extern "C" void run_mha(
    void *q_ptr,
    void *k_ptr,
    void *v_ptr,
    void *o_ptr,
    void *softmax_lse_ptr,
    void *alibi_slopes_ptr,

    int32_t *cu_seqlens_q_ptr,
    int32_t *cu_seqlens_k_ptr,

    uint32_t q_batch_stride,
    uint32_t k_batch_stride,
    uint32_t v_batch_stride,
    uint32_t o_batch_stride,
    uint32_t alibi_slopes_batch_stride,

    uint32_t q_row_stride,
    uint32_t k_row_stride,
    uint32_t v_row_stride,
    uint32_t o_row_stride,

    uint32_t q_head_stride,
    uint32_t k_head_stride,
    uint32_t v_head_stride,
    uint32_t o_head_stride,

    uint32_t b,
    uint32_t h,
    uint32_t h_k,
    uint32_t d,
    uint32_t d_rounded,
    float softmax_scale,

    uint32_t seqlen_q,
    uint32_t seqlen_k,
    uint32_t seqlen_q_rounded,
    uint32_t seqlen_k_rounded,

    int is_bf16,
    int is_causal,
    int unpadded_lse,

    int window_size_left,
    int window_size_right,

    float softcap,

    // SwarmLLM patch: the stream to launch on, handed in by the caller.
    // Upstream 0.10.x hardcoded `cudaStream_t stream = 0` below — see there.
    // Same shape as upstream's own fix (candle 0.11.0, PR #3655), so a bump
    // past 0.10 drops this patch rather than conflicting with it.
    void *stream_ptr
) {
    Flash_fwd_params params;
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q_ptr;
    params.k_ptr = k_ptr;
    params.v_ptr = v_ptr;
    params.o_ptr = o_ptr;

    params.softmax_lse_ptr = softmax_lse_ptr;
    params.alibi_slopes_ptr = alibi_slopes_ptr;

    // All stride are in elements, not bytes.
    params.q_batch_stride = q_batch_stride;
    params.k_batch_stride = k_batch_stride;
    params.v_batch_stride = v_batch_stride;
    params.o_batch_stride = o_batch_stride;
    params.alibi_slopes_batch_stride = alibi_slopes_batch_stride;

    params.q_row_stride = q_row_stride;
    params.k_row_stride = k_row_stride;
    params.v_row_stride = v_row_stride;
    params.o_row_stride = o_row_stride;
    params.q_head_stride = q_head_stride;
    params.k_head_stride = k_head_stride;
    params.v_head_stride = v_head_stride;
    params.o_head_stride = o_head_stride;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    params.p_dropout = 1.; // probability to keep
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    params.is_bf16 = is_bf16;
    params.cu_seqlens_q = cu_seqlens_q_ptr;
    params.cu_seqlens_k = cu_seqlens_k_ptr;
    params.p_ptr = nullptr; // used for `return_softmax`.
    params.seqused_k = nullptr;

    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
    params.num_splits = 1;
    params.unpadded_lse = unpadded_lse;

    // SwarmLLM patch: launch on the CALLER's stream, never a hardcoded one.
    //
    // Upstream wrote `cudaStream_t stream = 0; // Use the default stream.`,
    // which is correct only while candle itself runs on the legacy NULL
    // stream. The moment a `CudaDevice` owns a stream of its own — cudarc's
    // `new_stream()`, which is CU_STREAM_NON_BLOCKING and therefore does NOT
    // synchronise with stream 0 — this kernel ran unordered against every
    // kernel candle issued: it read Q/K/V before they were written and
    // candle read its output before it existed. That is what v0.3.199-alpha
    // shipped: every model emitted garbage on GPU (`给给给…`), because prefill
    // (always flash on CUDA) filled the KV cache with it.
    //
    // Measured on the published .200 artifact, one binary, env switches only:
    // own stream → garbage; own stream + CUDA_LAUNCH_BLOCKING=1 → byte-
    // identical to the legacy stream (ordering, not arithmetic); own stream
    // with flash replaced by standard attention → byte-identical to legacy +
    // standard. → `docs/invariants/inference.md` § "One CUDA stream per device".
    //
    // Upstream hit the same race and fixed it the same way: huggingface/candle
    // PR #3596 ("the attention kernels launch on a different stream than the
    // one that produced Q/K/V … a data race"), landed via PR #3655 in 0.11.0.
    //
    // "The types CUstream and cudaStream_t are identical and may be used
    // interchangeably" (CUDA Runtime API § Interactions with the Driver API),
    // so the driver handle cudarc gives us is the runtime stream. On the legacy
    // stream the caller hands in null — the default build launches exactly
    // where it always did.
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    run_mha_fwd(params, stream);
}
