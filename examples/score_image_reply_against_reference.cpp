// examples/score_against_reference.py for a picture: teacher-forces replies to an
// image prompt through llama.cpp (libmtmd, the stock build — Gemma 3's picture rows
// attend both ways, `mtmd_decode_use_non_causal`) and reports where each reply token
// ranks among llama.cpp's next-token choices, given the reply's own prefix.
//
//   ./mtmd_score MODEL.gguf MMPROJ.gguf IMAGE PROMPT_FILE REPLY_FILE...
//
// PROMPT_FILE holds the rendered chat prompt with `<__media__>` where the picture is
// (mtmd adds Gemma 3's <start_of_image>/<end_of_image> itself) and NO <bos> (added
// as add_special). Each REPLY_FILE holds one reply's text, scored as llama.cpp
// tokenizes it. Greedy on the CPU, n_ubatch 512 so a picture is one batch. Its
// "prompt positions" must equal the node's `usage.prompt_tokens`.
//
// Read the scores as score_against_reference.py's: llama.cpp's own reply is rank 1
// everywhere (the control); a reply from a correct implementation is rank 1 almost
// everywhere, each miss a near-tie (a fraction of a logit); a reply built on a
// different reading of the picture has misses several logits deep. #115 (2b),
// 2026-10-19: the node's both-ways replies 95.5% / 97.3% rank 1, worst gap 0.16 /
// 0.56; its earlier in-order replies 87.3% / 90.7%, gaps 12.0 / 5.7.
//
// Build against a llama.cpp checkout's CPU build (no cargo; not an [[example]]):
//   R=~/llama.cpp-ref; g++ -O2 -std=c++17 examples/score_image_reply_against_reference.cpp \
//     -I$R/include -I$R/ggml/include -I$R/tools/mtmd -L$R/build/bin \
//     -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$R/build/bin -o mtmd_score
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string slurp(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL.gguf MMPROJ.gguf IMAGE PROMPT_FILE REPLY_FILE...\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "cannot load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "cannot make context\n"); return 1; }

    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false;
    mcp.n_threads = 8;
    mcp.print_timings = false;
    mtmd_context * mctx = mtmd_init_from_file(argv[2], model, mcp);
    if (!mctx) { fprintf(stderr, "cannot load mmproj\n"); return 1; }

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "cannot read image\n"); return 1; }
    const std::string prompt = slurp(argv[4]);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int r = 5; r < argc; r++) {
        llama_memory_clear(llama_get_memory(lctx), true);
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        mtmd_input_text text = {prompt.c_str(), prompt.size(), true, true};
        const mtmd_bitmap * bitmaps[1] = {bw.bitmap};
        if (mtmd_tokenize(mctx, chunks, &text, bitmaps, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
        llama_pos n_past = 0;
        if (mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, 512, true, &n_past) != 0) {
            fprintf(stderr, "eval failed\n");
            return 1;
        }
        mtmd_input_chunks_free(chunks);

        const std::string reply = slurp(argv[r]);
        std::vector<llama_token> toks(reply.size() + 16);
        int n = llama_tokenize(vocab, reply.c_str(), reply.size(), toks.data(), toks.size(), false, false);
        if (n < 0) { fprintf(stderr, "reply tokenize failed\n"); return 1; }
        toks.resize(n);

        int rank1 = 0;
        float worst_gap = 0.f, min_p = 1.f;
        std::string misses;
        for (int i = 0; i < n; i++) {
            const float * lg = llama_get_logits_ith(lctx, -1);
            float mx = lg[0];
            for (int v = 1; v < n_vocab; v++) mx = std::max(mx, lg[v]);
            double z = 0;
            for (int v = 0; v < n_vocab; v++) z += std::exp((double) lg[v] - mx);
            const float l = lg[toks[i]];
            int rank = 1;
            int best = 0;
            for (int v = 0; v < n_vocab; v++) {
                if (lg[v] > l) rank++;
                if (lg[v] > lg[best]) best = v;
            }
            const float p = (float) (std::exp((double) l - mx) / z);
            const float gap = mx - l;
            if (rank == 1) rank1++;
            worst_gap = std::max(worst_gap, gap);
            min_p = std::min(min_p, p);
            if (rank > 1) {
                char piece[64], bpiece[64];
                int a = llama_token_to_piece(vocab, toks[i], piece, sizeof piece - 1, 0, true);
                int b = llama_token_to_piece(vocab, best, bpiece, sizeof bpiece - 1, 0, true);
                piece[a < 0 ? 0 : a] = 0;
                bpiece[b < 0 ? 0 : b] = 0;
                char line[256];
                snprintf(line, sizeof line, "    #%d %s rank %d p %.3f gap %.2f (llama.cpp: %s)\n",
                         i, piece, rank, p, gap, bpiece);
                misses += line;
            }
            llama_token t = toks[i];
            if (llama_decode(lctx, llama_batch_get_one(&t, 1)) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
            n_past++;
        }
        printf("%s: prompt positions %d, %d reply tokens, rank 1 at %d (%.1f%%), worst gap %.2f, min p %.3f\n%s",
               argv[r], (int) (n_past - n), n, rank1, 100.0 * rank1 / std::max(n, 1), worst_gap, min_p,
               misses.c_str());
        fflush(stdout);
    }
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
