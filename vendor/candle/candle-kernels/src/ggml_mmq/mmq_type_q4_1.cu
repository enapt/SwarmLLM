// SwarmLLM: llama.cpp's MMQ kernels for GGML_TYPE_Q4_1, one translation unit
// per type as llama.cpp's template-instances/mmq-instance-*.cu are (see swarm_mmq.cuh).
#include "swarm_mmq.cuh"

namespace swarm_ggml {
SWARM_MMQ_DECL(GGML_TYPE_Q4_1);
}
