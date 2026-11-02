// SwarmLLM: llama.cpp's MMQ kernels for GGML_TYPE_MXFP4 (gpt-oss's experts,
// FUTURE_WORK #390 step 7), one translation unit per type as llama.cpp's
// template-instances/mmq-instance-*.cu are (see swarm_mmq.cuh). The activation
// stays 8-bit (Q8_1, scale layout D4): the native FP4 tensor-core variant
// (`DECL_MMQ_CASE_W4A4`, Blackwell only) is not instantiated.
#include "swarm_mmq.cuh"

namespace swarm_ggml {
SWARM_MMQ_DECL(GGML_TYPE_MXFP4);
}
