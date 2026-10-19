// llama.cpp's reading of a picture, as the rows its image reader hands the
// language model — the reference `examples/image_rows_probe.rs` compares this
// node's rows against (FUTURE_WORK #310).
//
//   ./image_rows_reference MMPROJ.gguf IMAGE OUT.f32
//
// Runs llama.cpp's own preprocessor for the projector (`mtmd_image_preprocessor_*`,
// as `mtmd.cpp` picks it for a fixed-size encoder) and `clip_image_encode` on the
// CPU, then writes rows × width little-endian f32 and prints the shape. Needs only
// the mmproj: libmtmd's public API wants a text model as well, so this compiles
// llama.cpp's image-reader sources straight in (they are internal to libmtmd).
//
// Build against a llama.cpp checkout's CPU build (no cargo; not an [[example]]):
//   R=~/llama.cpp-ref; M=$R/tools/mtmd
//   g++ -O2 -std=c++17 examples/image_rows_reference.cpp $M/clip.cpp $M/mtmd-image.cpp \
//     $M/mtmd-audio.cpp $M/models/*.cpp -I$M -I$R/include -I$R/ggml/include -I$R/vendor \
//     -L$R/build/bin -lggml -lggml-base -lggml-cpu -lpthread -Wl,-rpath,$R/build/bin \
//     -o image_rows_reference
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "mtmd-image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <cstdio>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MMPROJ.gguf IMAGE OUT.f32\n", argv[0]);
        return 2;
    }
    clip_context_params params{};
    params.use_gpu = false;
    params.flash_attn_type = CLIP_FLASH_ATTN_TYPE_DISABLED;
    params.image_min_tokens = -1;
    params.image_max_tokens = -1;
    params.warmup = false;
    clip_init_result res = clip_init(argv[1], params);
    clip_ctx * ctx = res.ctx_v;
    if (!ctx) {
        fprintf(stderr, "could not load the image reader from %s\n", argv[1]);
        return 1;
    }

    int w = 0, h = 0, n = 0;
    unsigned char * px = stbi_load(argv[2], &w, &h, &n, 3);
    if (!px) {
        fprintf(stderr, "could not read %s\n", argv[2]);
        return 1;
    }
    clip_image_u8 img;
    img.set_size({w, h}, false);
    img.cpy_buf(std::vector<uint8_t>(px, px + (size_t) w * h * 3));
    stbi_image_free(px);

    // The preprocessor mtmd.cpp chooses for LLaVA without pinpoints and for
    // Gemma 3: one fixed-size square.
    mtmd_image_preprocessor_fixed_size pre(ctx);
    mtmd_image_preproc_out out = pre.preprocess(img);
    if (out.entries.size() != 1) {
        fprintf(stderr, "expected one preprocessed image, got %zu\n", out.entries.size());
        return 1;
    }
    const clip_image_f32 & f = out.entries[0];
    const int rows = clip_n_output_tokens(ctx, &f);
    const int width = clip_n_mmproj_embd(ctx);
    std::vector<float> emb((size_t) rows * width);
    if (!clip_image_encode(ctx, 4, &f, emb)) {
        fprintf(stderr, "encode failed\n");
        return 1;
    }
    FILE * fo = fopen(argv[3], "wb");
    if (!fo || fwrite(emb.data(), sizeof(float), emb.size(), fo) != emb.size()) {
        fprintf(stderr, "could not write %s\n", argv[3]);
        return 1;
    }
    fclose(fo);
    printf("rows=%d width=%d image=%dx%d preprocessed=%dx%d\n", rows, width, w, h, f.nx(), f.ny());
    clip_free(ctx);
    return 0;
}
