// EXP-009: minimal dump → policy-mask → writeback → continue prefill for GUI KV.
// Links against packaged libllama.so (Android aarch64 / Hexagon pkg).
// Does NOT modify llama.cpp core. Merge writeback of synthetic cells is stubbed;
// keep/drop uses llama_memory_seq_rm. NPU DMA remains stub (host memcpy only).
//
// Usage (on device):
//   LD_LIBRARY_PATH=lib ADSP_LIBRARY_PATH=lib GGML_HEXAGON_EXPERIMENTAL=1 \
//   ./bin/kv_gui_e2e -m MODEL -c 2048 --device HTP0 -ngl 99 \
//     --dump-dir /data/local/tmp/exp009 --keep-mask keep_mask.txt \
//     [--image next.png] [-p "next prompt"]
//
// Modes via env/flags:
//   --phase dump     : multi-segment prefill, write state + meta + features
//   --phase apply    : load state, apply keep_mask (seq_rm drop), save state
//   --phase prefill  : load state, prefill next prompt/image, report timings
//   --phase e2e      : dump → (external policy) skip → if mask present apply → prefill
//                      For full e2e with policy on host: dump, pull, policy, push mask+apply, prefill.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct SegSpec {
    std::string id;
    std::string kind; // task | action_prev | history
    std::string text;
};

// Default multi-turn GUI-like text history (no image required for dump of LLM KV).
static std::vector<SegSpec> default_segments() {
    return {
        {"task", "task",
         "Task: Open Settings and enable Wi-Fi. Stay on the current app context."},
        {"history_0", "history",
         "Step0: Home screen visible. Icons: Settings, Camera, Browser."},
        {"history_1", "history",
         "Step1: Tapped Settings icon at bottom left. Transitioning."},
        {"history_2", "history",
         "Step2: Settings list shows Wi-Fi, Bluetooth, Display."},
        {"history_3", "history",
         "Step3: Scrolled down past Network section once."},
        {"history_4", "history",
         "Step4: Saw unrelated About phone entry; ignore for task."},
        {"history_5", "history",
         "Step5: Returned focus toward Wireless & networks."},
        {"action_prev", "action_prev",
         "Previous action: tap(x=540,y=820) on Wi-Fi row. Waiting for toggle UI."},
    };
}

static bool write_text(const std::string & path, const std::string & s) {
    std::ofstream f(path);
    if (!f) return false;
    f << s;
    return (bool)f;
}

static bool read_keep_mask(const std::string & path, std::vector<int> & keep) {
    std::ifstream f(path);
    if (!f) return false;
    keep.clear();
    int v;
    while (f >> v) keep.push_back(v);
    return !keep.empty();
}

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int & n_past, int n_batch) {
    const int n = (int)tokens.size();
    for (int i = 0; i < n; i += n_batch) {
        const int n_eval = std::min(n_batch, n - i);
        llama_batch batch = llama_batch_init(n_eval, 0, 1);
        for (int j = 0; j < n_eval; ++j) {
            common_batch_add(batch, tokens[i + j], n_past + j, {0}, j == n_eval - 1);
        }
        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            fprintf(stderr, "decode failed at n_past=%d\n", n_past);
            return false;
        }
        llama_batch_free(batch);
        n_past += n_eval;
    }
    return true;
}

// Dump a compact feature row per cell: mean of layer-0..L-1 K rows is NOT available via public API
// without reading tensors. We instead dump:
//   - session state file (full KV)
//   - seq state blob
//   - meta JSON with segment token ranges + sha256 of seq blob
// Host-side Python parser extracts per-token K features from seq state (documented layout).
static bool dump_seq_blob(llama_context * ctx, const std::string & path, size_t & nbytes) {
    const size_t sz = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> buf(sz);
    const size_t n = llama_state_seq_get_data(ctx, buf.data(), buf.size(), 0);
    if (n != sz) {
        fprintf(stderr, "seq get_data size mismatch %zu vs %zu\n", n, sz);
        return false;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char *>(buf.data()), (std::streamsize)buf.size());
    nbytes = buf.size();
    return (bool)f;
}

static int apply_keep_mask(llama_context * ctx, const std::vector<int> & keep, int n_pos) {
    // keep[i]==1 means keep position i; drop contiguous runs of 0 via seq_rm
    llama_memory_t mem = llama_get_memory(ctx);
    int dropped = 0;
    int i = 0;
    while (i < n_pos) {
        if (keep[i]) { ++i; continue; }
        int j = i;
        while (j < n_pos && !keep[j]) ++j;
        // remove [i, j)
        if (!llama_memory_seq_rm(mem, 0, i, j)) {
            fprintf(stderr, "seq_rm failed for [%d,%d)\n", i, j);
            return -1;
        }
        dropped += (j - i);
        // After rm, positions after j shift? llama_memory_seq_rm removes cells but positions
        // of remaining cells stay as their pos values; holes may remain. For subsequent
        // decode, llama typically allows non-contiguous pos. We keep original pos ids in mask space.
        i = j;
    }
    return dropped;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 16;
    params.sampling.temp = 0.0f;

    std::string phase = "dump";
    std::string dump_dir = "/data/local/tmp/exp009";
    std::string keep_mask_path;
    std::string next_prompt = "Next observation: Wi-Fi toggle is now visible. What is the next action?";
    bool have_custom_prompt = false;

    // Pre-parse custom flags then let common_params_parse handle the rest.
    std::vector<char *> filtered;
    filtered.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--phase" && i + 1 < argc) { phase = argv[++i]; continue; }
        if (a == "--dump-dir" && i + 1 < argc) { dump_dir = argv[++i]; continue; }
        if (a == "--keep-mask" && i + 1 < argc) { keep_mask_path = argv[++i]; continue; }
        if (a == "--next-prompt" && i + 1 < argc) { next_prompt = argv[++i]; have_custom_prompt = true; continue; }
        filtered.push_back(argv[i]);
    }
    int fargc = (int)filtered.size();
    char ** fargv = filtered.data();

    common_init();
    if (!common_params_parse(fargc, fargv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    const int n_batch = params.n_batch;
    const std::string state_path = dump_dir + "/kv_state.bin";
    const std::string seq_path = dump_dir + "/kv_seq0.bin";
    const std::string meta_path = dump_dir + "/segments.json";
    const std::string timing_path = dump_dir + "/timings.json";

    // Ensure dir exists (best-effort; caller should mkdir)
    {
        std::string cmd = "mkdir -p '" + dump_dir + "'";
        system(cmd.c_str());
    }

    std::vector<SegSpec> segs = default_segments();
    std::ostringstream meta;
    meta << "{\n  \"model\": \"from-cli\",\n  \"segments\": [\n";

    int n_past = 0;
    std::vector<llama_token> all_tokens;
    double t_prefill0 = now_ms();

    if (phase == "dump" || phase == "e2e") {
        for (size_t si = 0; si < segs.size(); ++si) {
            auto & seg = segs[si];
            // Separate segments with a clear marker line so tokenization ranges are stable.
            std::string piece = std::string("[SEG ") + seg.id + "/" + seg.kind + "]\n" + seg.text + "\n";
            auto toks = common_tokenize(ctx, piece, /*add_special=*/(si == 0), true);
            const int pos0 = n_past;
            if (!decode_tokens(ctx, toks, n_past, n_batch)) return 1;
            const int pos1 = n_past;
            all_tokens.insert(all_tokens.end(), toks.begin(), toks.end());
            meta << "    {\"seg_id\":\"" << seg.id << "\",\"kind\":\"" << seg.kind
                 << "\",\"pos0\":" << pos0 << ",\"pos1\":" << pos1
                 << ",\"n_tokens\":" << (pos1 - pos0) << "}"
                 << (si + 1 < segs.size() ? "," : "") << "\n";
            fprintf(stderr, "seg %s kind=%s tokens=%d pos=[%d,%d)\n",
                    seg.id.c_str(), seg.kind.c_str(), pos1 - pos0, pos0, pos1);
        }
        meta << "  ],\n  \"n_tokens_total\":" << n_past << ",\n";
        double t_prefill1 = now_ms();

        // Save full session + seq blob
        if (!llama_state_save_file(ctx, state_path.c_str(), all_tokens.data(), all_tokens.size())) {
            fprintf(stderr, "state_save_file failed\n");
            return 1;
        }
        size_t seq_bytes = 0;
        if (!dump_seq_blob(ctx, seq_path, seq_bytes)) {
            fprintf(stderr, "dump seq failed\n");
            return 1;
        }
        meta << "  \"state_path\":\"" << state_path << "\",\n";
        meta << "  \"seq_path\":\"" << seq_path << "\",\n";
        meta << "  \"seq_bytes\":" << seq_bytes << ",\n";
        meta << "  \"prefill_history_ms\":" << (t_prefill1 - t_prefill0) << ",\n";
        meta << "  \"n_pos\":" << n_past << "\n}\n";
        if (!write_text(meta_path, meta.str())) {
            fprintf(stderr, "write meta failed\n");
            return 1;
        }
        fprintf(stderr, "dump ok: n_pos=%d seq_bytes=%zu state=%s\n", n_past, seq_bytes, state_path.c_str());

        if (phase == "dump") {
            // Also write a trivial all-keep mask template
            std::ostringstream mask;
            for (int i = 0; i < n_past; ++i) mask << "1\n";
            write_text(dump_dir + "/keep_mask_all.txt", mask.str());
            return 0;
        }
    }

    // For apply / prefill / e2e-with-mask: load state if we did not just dump in same process.
    if (phase == "apply" || phase == "prefill") {
        std::vector<llama_token> toks_out(1 << 16);
        size_t n_tok_out = 0;
        if (!llama_state_load_file(ctx, state_path.c_str(), toks_out.data(), toks_out.size(), &n_tok_out)) {
            fprintf(stderr, "state_load_file failed: %s\n", state_path.c_str());
            return 1;
        }
        n_past = (int)n_tok_out;
        all_tokens.assign(toks_out.begin(), toks_out.begin() + n_tok_out);
        fprintf(stderr, "loaded state tokens=%zu pos_max=%d\n", n_tok_out,
                (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0));
    }

    if ((phase == "apply" || phase == "e2e") && !keep_mask_path.empty()) {
        std::vector<int> keep;
        if (!read_keep_mask(keep_mask_path, keep)) {
            fprintf(stderr, "failed to read keep mask %s\n", keep_mask_path.c_str());
            return 1;
        }
        // n_pos from mask length
        const int n_pos = (int)keep.size();
        double t0 = now_ms();
        int dropped = apply_keep_mask(ctx, keep, n_pos);
        double t1 = now_ms();
        if (dropped < 0) return 1;
        const std::string state_out = dump_dir + "/kv_state_compressed.bin";
        // tokens list after drop: keep only masked tokens (order preserved)
        std::vector<llama_token> kept_toks;
        for (int i = 0; i < n_pos && i < (int)all_tokens.size(); ++i) {
            if (keep[i]) kept_toks.push_back(all_tokens[i]);
        }
        if (!llama_state_save_file(ctx, state_out.c_str(), kept_toks.data(), kept_toks.size())) {
            fprintf(stderr, "save compressed state failed\n");
            return 1;
        }
        size_t seq_bytes = 0;
        dump_seq_blob(ctx, dump_dir + "/kv_seq0_compressed.bin", seq_bytes);
        std::ostringstream tj;
        tj << "{\"apply_ms\":" << (t1 - t0) << ",\"dropped\":" << dropped
           << ",\"kept\":" << (n_pos - dropped) << ",\"seq_bytes_compressed\":" << seq_bytes << "}\n";
        write_text(dump_dir + "/apply_stats.json", tj.str());
        fprintf(stderr, "apply ok dropped=%d kept=%d apply_ms=%.3f\n", dropped, n_pos - dropped, t1 - t0);
        if (phase == "apply") return 0;
        // e2e continues to prefill with compressed KV already in ctx
    }

    if (phase == "prefill" || phase == "e2e") {
        // If phase prefill and compressed state exists, prefer it
        if (phase == "prefill") {
            const std::string state_c = dump_dir + "/kv_state_compressed.bin";
            std::ifstream test(state_c);
            if (test.good()) {
                llama_memory_clear(llama_get_memory(ctx), true);
                std::vector<llama_token> toks_out(1 << 16);
                size_t n_tok_out = 0;
                if (!llama_state_load_file(ctx, state_c.c_str(), toks_out.data(), toks_out.size(), &n_tok_out)) {
                    fprintf(stderr, "load compressed failed\n");
                    return 1;
                }
                n_past = (int)llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
                fprintf(stderr, "loaded compressed state n_tok=%zu n_past_est=%d\n", n_tok_out, n_past);
            }
        }

        auto next_toks = common_tokenize(ctx, next_prompt, false, true);
        double t0 = now_ms();
        if (!decode_tokens(ctx, next_toks, n_past, n_batch)) return 1;
        double t1 = now_ms();

        // sample a few tokens for sanity
        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        std::string gen;
        llama_batch batch = llama_batch_init(1, 0, 1);
        double t2 = now_ms();
        for (int i = 0; i < params.n_predict; ++i) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            gen += common_token_to_piece(ctx, tok);
            if (llama_vocab_is_eog(llama_model_get_vocab(model), tok)) break;
            common_batch_clear(batch);
            common_batch_add(batch, tok, n_past, {0}, true);
            if (llama_decode(ctx, batch)) break;
            n_past += 1;
        }
        double t3 = now_ms();
        llama_batch_free(batch);
        llama_sampler_free(smpl);

        std::ostringstream tj;
        tj << "{\"next_prompt_tokens\":" << next_toks.size()
           << ",\"prefill_next_ms\":" << (t1 - t0)
           << ",\"decode_ms\":" << (t3 - t2)
           << ",\"n_predict\":" << params.n_predict
           << ",\"gen\":\"" << gen << "\"}\n";
        write_text(dump_dir + "/prefill_stats.json", tj.str());
        write_text(dump_dir + "/generation.txt", gen);
        fprintf(stderr, "prefill_next_ms=%.3f decode_ms=%.3f gen=%s\n", t1 - t0, t3 - t2, gen.c_str());
    }

    return 0;
}
